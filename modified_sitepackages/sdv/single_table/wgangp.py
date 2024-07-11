# WPGAN module
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.optim as optim
from torch.utils.data import DataLoader

from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

from tqdm import tqdm
import wandb

def _validate_no_category_dtype(data):
    """Check that given data has no 'category' dtype columns.

    Args:
        data (pd.DataFrame):
            Data to check.

    Raises:
        - ``InvalidDataTypeError`` if any columns in the data have 'category' dtype.
    """
    category_cols = [
        col for col, dtype in data.dtypes.items() if pd.api.types.is_categorical_dtype(dtype)
    ]
    if category_cols:
        categoricals = "', '".join(category_cols)
        error_msg = (
            f"Columns ['{categoricals}'] are stored as a 'category' type, which is not "
            "supported. Please cast these columns to an 'object' to continue."
        )
        raise InvalidDataTypeError(error_msg)

class Generator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()

        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item
        seq += [nn.Linear(dim, data_dim)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data
class Discriminator(nn.Module):
    def __init__(self, embedding_dim, discriminator_dim, pac=1):
        super(Discriminator, self).__init__()
        dim = embedding_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item
        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)
        self.sig = nn.Sigmoid()

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=1, lambda_=10):
        """Compute the gradient penalty."""
        real_data.to(device)
        fake_data.to(device)

        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1))
        gradients_norm = torch.sqrt(torch.sum(gradients_view ** 2, dim=1) + 1e-12)
        gradient_penalty = lambda_ * ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty

    def forward(self, input_, return_logits = False):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        logits = self.seq(input_.view(-1, self.pacdim))
        if return_logits:
            return self.sig(logits), logits
        else:
            return self.sig(logits)


class WGANGP(BaseSynthesizer):
    def __init__(self, embedding_dim=30, generator_dim=(128, 256, 512), discriminator_dim=(512, 256, 128),
                 generator_lr=2e-4, generator_decay=0, discriminator_lr=2e-4, discriminator_decay=0,
                 discriminator_stored= False, betas= (.9, .99), batch_size=64, discriminator_steps=5,
                 log_frequency=True, verbose=False, epochs= 4000, pac=10, cuda=True, use_wandb= False):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._discriminator_stored = discriminator_stored

        self._betas = betas

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._use_wandb = use_wandb
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')
    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(embedding_dim= self._embedding_dim, generator_dim= self._generator_dim, data_dim= data_dim).to(self._device)

        if self._discriminator_stored:
            self._discriminator =  Discriminator(embedding_dim= data_dim, discriminator_dim= self._discriminator_dim).to(self._device)
        else:
            discriminator = Discriminator(embedding_dim= data_dim, discriminator_dim= self._discriminator_dim).to(self._device)

        optimizerG = optim.Adam(self._generator.parameters(), lr=self._generator_lr, betas= self._betas, weight_decay=self._generator_decay)

        if self._discriminator_stored:
            optimizerD = optim.Adam(self._discriminator.parameters(), lr=self._discriminator_lr, betas= self._betas, weight_decay=self._discriminator_decay)
        else:
            optimizerD = optim.Adam(discriminator.parameters(), lr=self._discriminator_lr, betas=self._betas, weight_decay=self._discriminator_decay)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])
        self.num_steps = 0

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                real = self._data_sampler.sample_data(train_data, self._batch_size, None, None)
                real = torch.from_numpy(real.astype('float32')).to(self._device)

                for n in range(self._discriminator_steps):
                    optimizerG.zero_grad()

                    # Get generated data
                    fakez = torch.normal(mean=mean, std=std)
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    # Calculate loss and optimize
                    if self._discriminator_stored:
                        d_generated = self._discriminator(fakeact)
                    else:
                        d_generated = discriminator(fakeact)
                    loss_g = - d_generated.mean()
                    loss_g.backward()
                    optimizerG.step()

            # Get generated data
            fakez = torch.normal(mean=mean, std=std)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            # Calculate probabilities on real and generated data
            if self._discriminator_stored:
                d_real = self._discriminator(real)
                d_generated = self._discriminator(fakeact)

                # Get gradient penalty
                gradient_penalty = self._discriminator.calc_gradient_penalty(real, fakeact, device=self._device)
            else:
                d_real = discriminator(real)
                d_generated = discriminator(fakeact)

                # Get gradient penalty
                gradient_penalty = discriminator.calc_gradient_penalty(real, fakeact, device=self._device)

            # Create total loss and optimize
            optimizerD.zero_grad()
            loss_d = d_generated.mean() - d_real.mean() + gradient_penalty
            loss_d.backward()

            optimizerD.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            if self._use_wandb:
                wandb.log({"Generator Loss": generator_loss, "Discriminator Loss": discriminator_loss})

    @random_state
    def sample(self, n):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1

            fakez = torch.normal(mean=mean, std=std).to(self._device)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
        if self._discriminator_stored and self._discriminator is not None:
            self._discriminator.to(self._device)

class WGANGP_DRS(WGANGP):
    def __init__(self, embedding_dim=30, generator_dim=(128, 256, 512), discriminator_dim=(512, 256, 128),
                 generator_lr=2e-4, generator_decay=0, discriminator_lr=2e-4, burnin_samples= 50000,
                 discriminator_decay=0, betas=(.9, .99), batch_size=64, discriminator_steps=5, dsr_epsilon= 1e-8, dsr_gamma_percentile= 0.80,
                 log_frequency=True, verbose=False, epochs=4000, pac=10, cuda=True, use_wandb= False):
        super().__init__(embedding_dim=embedding_dim, generator_dim=generator_dim, discriminator_dim=discriminator_dim,
                 generator_lr=generator_lr, generator_decay=generator_decay, discriminator_lr=discriminator_lr,
                 discriminator_decay=discriminator_decay, discriminator_stored= True, betas=betas, batch_size=batch_size, discriminator_steps=discriminator_steps,
                 log_frequency=log_frequency, verbose=verbose, epochs=epochs, pac=pac, cuda=cuda, use_wandb=use_wandb)

        self._burnin_samples = burnin_samples
        self._burnin_done = False
        self._dsr_epsilon = dsr_epsilon
        self._dsr_gamma_percentile = dsr_gamma_percentile

        self.max_M = 0.0
        self.max_logit = 0.0

    @random_state
    def sample(self, n, do_burnin= False):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        # BurnIn
        if not self._burnin_done or do_burnin:
            steps_burnin = max(self._burnin_samples // self._batch_size, 1)
            burnin_iterator = tqdm(range(steps_burnin), disable=(not self._verbose))

            if self._verbose:
                description = 'DSR Burnin'
                burnin_iterator.set_description(description.format(gen=0, dis=0))

            for i in burnin_iterator:
                mean = torch.zeros(self._batch_size, self._embedding_dim)
                std = mean + 1

                fakez = torch.normal(mean=mean, std=std).to(self._device)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                generated_d, logits = self._discriminator(fakeact, return_logits= True)

                batch_ratio = torch.exp(logits)
                max_idx = torch.argmax(batch_ratio)
                max_ratio = batch_ratio[max_idx].detach().cpu().numpy()[0]

                if max_ratio > self.max_M:
                    self.max_M = max_ratio
                    self.max_logit = logits[max_idx].detach().cpu().numpy()[0]

            self._burnin_done = True

        # Sample
        data = []
        generated_samples = 0
        while generated_samples < n:
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1

            fakez = torch.normal(mean=mean, std=std).to(self._device)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            # postprocess with Discriminator Rejection Sampling
            generated_d, logits = self._discriminator(fakeact, return_logits= True)
            batch_ratio = torch.exp(logits)
            max_idx = torch.argmax(batch_ratio)
            max_ratio = batch_ratio[max_idx].detach().cpu().numpy()[0]
            # update max_M if larger M is found
            if max_ratio > self.max_M:
                self.max_M = max_ratio
                self.max_logit = logits[max_idx].detach().cpu().numpy()[0]

            # calculate F_hat and pass it into sigmoid
            # set gamma dynamically (95th percentile of F)
            logits = logits.detach().cpu().numpy()
            Fs = logits - self.max_logit - np.log(1 - np.exp(logits - self.max_logit - self._dsr_epsilon))
            gamma = np.percentile(Fs, self._dsr_gamma_percentile)
            F_hat = Fs - gamma
            acceptance_prob = 1/(1 + np.exp(-F_hat))
            probability = np.random.uniform(0, 1, size= acceptance_prob.shape)

            fakeact = fakeact.detach().cpu().numpy()[(acceptance_prob > probability)[:, 0]]

            data.append(fakeact)
            generated_samples += fakeact.shape[0]

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

class WGANGPSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``WGANGP`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.

        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 locales=['en_US'],  embedding_dim=30, generator_dim=(128, 256, 512), discriminator_dim=(512, 256, 128),
                 generator_lr=2e-4, generator_decay=0, discriminator_lr=2e-4, discriminator_decay=0, betas= (.9, .99), batch_size=64, discriminator_steps=5,
                 log_frequency=True, verbose=False, epochs= 4000, pac=10, cuda=True, use_wandb= False):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.betas = betas
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.cuda = cuda
        self.use_wandb = use_wandb

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'betas': betas,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'use_wandb': use_wandb
        }

    def _estimate_num_columns(self, data):
        """Estimate the number of columns that the data will generate.

        Estimates that continuous columns generate 11 columns and categorical ones
        create n where n is the number of unique categories.

        Args:
            data (pandas.DataFrame):
                Data to estimate the number of columns from.

        Returns:
            int:
                Number of estimate columns.
        """
        sdtypes = self._data_processor.get_sdtypes()
        transformers = self.get_transformers()
        num_generated_columns = {}
        for column in data.columns:
            if column not in sdtypes:
                continue

            if sdtypes[column] in {'numerical', 'datetime'}:
                num_generated_columns[column] = 11

            elif sdtypes[column] in {'categorical', 'boolean'}:
                if transformers.get(column) is None:
                    num_categories = data[column].fillna(np.nan).nunique(dropna=False)
                    num_generated_columns[column] = num_categories
                else:
                    num_generated_columns[column] = 11

        return num_generated_columns

    def _print_warning(self, data):
        """Print a warning if the number of columns generated is over 1000."""
        dict_generated_columns = self._estimate_num_columns(data)
        if sum(dict_generated_columns.values()) > 1000:
            header = {'Original Column Name  ': 'Est # of Columns (WGANGP)'}
            dict_generated_columns = {**header, **dict_generated_columns}
            longest_column_name = len(max(dict_generated_columns, key=len))
            cap = '<' + str(longest_column_name)
            lines_to_print = []
            for column, num_generated_columns in dict_generated_columns.items():
                lines_to_print.append(f'{column:{cap}} {num_generated_columns}')

            generated_columns_str = '\n'.join(lines_to_print)
            print(  # noqa: T001
                'PerformanceAlert: Using the WGANGPSynthesizer on this data is not recommended. '
                'To model this data, WGANGP will generate a large number of columns.'
                '\n\n'
                f'{generated_columns_str}'
                '\n\n'
                'We recommend preprocessing discrete columns that can have many values, '
                "using 'update_transformers'. Or you may drop columns that are not necessary "
                'to model. (Exit this script using ctrl-C)'
            )

    def _preprocess(self, data):
        self.validate(data)
        self._data_processor.fit(data)
        self._print_warning(data)

        return self._data_processor.transform(data)

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        _validate_no_category_dtype(processed_data)

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            processed_data,
            transformers
        )

        self._model = WGANGP(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("WGANGPSynthesizer doesn't support conditional sampling.")

class WGANGP_DRSSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``WGANGP_DRS`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.

        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 locales=['en_US'],  embedding_dim=30, generator_dim=(128, 256, 512), discriminator_dim=(512, 256, 128),
                 generator_lr=2e-4, generator_decay=0, discriminator_lr=2e-4, discriminator_decay=0, betas= (.9, .99), batch_size=64, discriminator_steps=5,
                 log_frequency=True, verbose=False, epochs= 4000, pac=10, dsr_epsilon= 1e-8, dsr_gamma_percentile= 0.80, cuda=True, use_wandb= False):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.betas = betas
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.dsr_epsilon = dsr_epsilon
        self.dsr_gamma_percentile = dsr_gamma_percentile
        self.cuda = cuda
        self.use_wandb = use_wandb

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'betas': betas,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'dsr_epsilon': dsr_epsilon,
            'dsr_gamma_percentile': dsr_gamma_percentile,
            'cuda': cuda,
            'use_wandb': use_wandb
        }

    def _estimate_num_columns(self, data):
        """Estimate the number of columns that the data will generate.

        Estimates that continuous columns generate 11 columns and categorical ones
        create n where n is the number of unique categories.

        Args:
            data (pandas.DataFrame):
                Data to estimate the number of columns from.

        Returns:
            int:
                Number of estimate columns.
        """
        sdtypes = self._data_processor.get_sdtypes()
        transformers = self.get_transformers()
        num_generated_columns = {}
        for column in data.columns:
            if column not in sdtypes:
                continue

            if sdtypes[column] in {'numerical', 'datetime'}:
                num_generated_columns[column] = 11

            elif sdtypes[column] in {'categorical', 'boolean'}:
                if transformers.get(column) is None:
                    num_categories = data[column].fillna(np.nan).nunique(dropna=False)
                    num_generated_columns[column] = num_categories
                else:
                    num_generated_columns[column] = 11

        return num_generated_columns

    def _print_warning(self, data):
        """Print a warning if the number of columns generated is over 1000."""
        dict_generated_columns = self._estimate_num_columns(data)
        if sum(dict_generated_columns.values()) > 1000:
            header = {'Original Column Name  ': 'Est # of Columns (WGANGP)'}
            dict_generated_columns = {**header, **dict_generated_columns}
            longest_column_name = len(max(dict_generated_columns, key=len))
            cap = '<' + str(longest_column_name)
            lines_to_print = []
            for column, num_generated_columns in dict_generated_columns.items():
                lines_to_print.append(f'{column:{cap}} {num_generated_columns}')

            generated_columns_str = '\n'.join(lines_to_print)
            print(  # noqa: T001
                'PerformanceAlert: Using the WGANGPSynthesizer on this data is not recommended. '
                'To model this data, WGANGP will generate a large number of columns.'
                '\n\n'
                f'{generated_columns_str}'
                '\n\n'
                'We recommend preprocessing discrete columns that can have many values, '
                "using 'update_transformers'. Or you may drop columns that are not necessary "
                'to model. (Exit this script using ctrl-C)'
            )

    def _preprocess(self, data):
        self.validate(data)
        self._data_processor.fit(data)
        self._print_warning(data)

        return self._data_processor.transform(data)

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        _validate_no_category_dtype(processed_data)

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            processed_data,
            transformers
        )

        self._model = WGANGP_DRS(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("WGANGPSynthesizer doesn't support conditional sampling.")
