from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns

import warnings
from collections import namedtuple
from joblib import Parallel, delayed

# import data science libraries
import pandas as pd
import numpy as np
import math

# import scikit-learn preprocessing
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

# import pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ctgan.synthesizers.base import BaseSynthesizer, random_state

# import synthetic data vault libraries
from sdv.metadata import SingleTableMetadata
import sdv.evaluation.single_table as sdv_st

# import utility libraries
from tqdm import tqdm
import wandb
from datetime import datetime

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

# define base feedforward network
class BaseNetwork(nn.Module):

    # define base network constructor
    def __init__(self, hidden_size, activation='lrelu'):

        # call super calass constructor
        super(BaseNetwork, self).__init__()

        # init
        self.layers = self.init_layers(hidden_size)

        # case: lrelu activation
        if activation == 'lrelu':

            # set lrelu activation
            self.activation = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: relu activation
        elif activation == 'relu':

            # set relu activation
            self.activation = nn.ReLU(inplace=True)

        # case: tanh activation
        elif activation == 'tanh':

            # set tanh activation
            self.activation = nn.Tanh()

        # case: sigmoid activation
        else:

            # set sigmoid activation
            self.activation = nn.Sigmoid()

    # define layer initialization
    def init_layers(self, layer_dimensions):

        # init layers
        layers = []

        # iterate over layer dimensions
        for i in range(len(layer_dimensions) - 1):
            # init linear layer
            layer = nn.Linear(layer_dimensions[i], layer_dimensions[i + 1], bias=True)

            # init linear layer weights
            nn.init.xavier_uniform_(layer.weight)

            # init linear layer bias
            nn.init.constant_(layer.bias, 0.0)

            # collecet linear layer
            layers.append(layer)

            # register linear layer parameters
            self.add_module('linear_' + str(i), layer)

        # return layers
        return layers

    # define forward pass
    def forward(self, x):

        # iterate over layers
        for i in range(len(self.layers)):
            # run layer forward pass
            x = self.activation(self.layers[i](x))

        # return forward pass result
        return x

# define MLP synthesizer network
class MLPSynthesizer(nn.Module):

    # define MLP synthesizer network constructor
    def __init__(
            self,
            d_in: int,
            hidden_layers: list,
            activation: str = 'lrelu',  # layer activation
            dim_t: int = 64,
            n_cat_tokens=None,  # number of categorical tokens
            n_cat_emb=None,  # number of categorical dimensions
            embedding=None,
            embedding_learned=True,
            n_classes=None
    ):

        # call super class constructor
        super(MLPSynthesizer, self).__init__()

        # init ???
        self.dim_t = dim_t

        # init synthesizer base feed forward network
        self.backbone = BaseNetwork([dim_t, *hidden_layers], activation=activation)

        # case: categorical embedding defined
        if embedding is not None:

            # init pretrained embedding layer
            self.cat_embedding = nn.Embedding.from_pretrained(embeddings=embedding)

        # case: categorical embedding undefined
        else:

            # init new categorical embedding layer
            self.cat_embedding = nn.Embedding(n_cat_tokens, n_cat_emb, max_norm=None, scale_grad_by_freq=False)

            # activate categorical embedding layer learning
            self.cat_embedding.weight.requires_grad = embedding_learned

        # case: data classes available
        if n_classes is not None:
            # init label embedding layer
            self.label_embedding = nn.Embedding(n_classes, dim_t)

        # define input data projection
        self.projection = nn.Sequential(
            nn.Linear(d_in, dim_t),  # linear layer
            nn.SiLU(),  # silu activation
            nn.Linear(dim_t, dim_t)  # linear layer
        )

        # define time embedding projection
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),  # linear layer
            nn.SiLU(),  # silu activation
            nn.Linear(dim_t, dim_t)  # linear layer
        )

        # define output data projection
        self.head = nn.Linear(hidden_layers[-1], d_in)

    # define sinusodial time step embedding
    def embed_time(self, timesteps, dim_out, max_period=10000):

        # half output dimension
        half_dim_out = dim_out // 2

        # determine tensor of frequencies
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim_out, dtype=torch.float32) / half_dim_out)

        # push to compute device
        freqs = freqs.to(device=timesteps.device)

        # create timestep vs. frequency grid
        args = timesteps[:, None].float() * freqs[None]

        # creating the time embedding
        time_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # case: odd output dimension
        if dim_out % 2:
            # append additional dimension
            time_embedding = torch.cat([time_embedding, torch.zeros_like(time_embedding[:, :1])], dim=-1)

        # return timestep embedding
        return time_embedding

    # get categorical embeddings
    def get_embeddings(self):

        # return categorical embeddings
        return self.cat_embedding.weight.data

    # perform categorical embedding
    def embed_categorical(self, x_cat):

        # perform categorical embedding
        x_cat_emb = self.cat_embedding(x_cat)

        # reshape embedding to original input
        x_cat_emb = x_cat_emb.view(-1, x_cat_emb.shape[1] * x_cat_emb.shape[2])

        # return categorical embedding
        return x_cat_emb

    # define forward pass
    def forward(self, x, timesteps, label=None):

        # init time embeddings
        time_emb = self.embed_time(timesteps, self.dim_t)

        # embedd time embeddings
        time_emb = self.time_embed(time_emb)

        # case: data classes available
        if label is not None:
            # determine label embeddings
            time_label_emb = time_emb + self.label_embedding(label)
            # add time and label embedding
            x = x + time_label_emb

        # run initial projection layer
        x = self.projection(x)

        # run backbone forward pass
        x = self.backbone(x)

        # run projection forward pass
        x = self.head(x)

        # return forward pass result
        return x

# define BaseDiffuser network
class BaseDiffuser(object):

    # define base diffuser network constructor
    def __init__(
            self,
            total_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device='cpu',
            scheduler='linear'
    ):

        # set diffusion steps
        self.total_steps = total_steps

        # set diffusion start beta
        self.beta_start = beta_start

        # set diffusion end beta
        self.beta_end = beta_end

        # set compute device
        self.device = device

        # set noise schedule alphas and betas
        self.alphas, self.betas = self.prepare_noise_schedule(scheduler=scheduler)

        # set noise schedule alhpa hats
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    # define noise schedule
    def prepare_noise_schedule(self, scheduler: str):

        # determine noise scheduler scale
        scale = 1000 / self.total_steps

        # scale beta start
        beta_start = scale * self.beta_start

        # scale beta end
        beta_end = scale * self.beta_end

        # case: linear noise scheduler
        if scheduler == 'linear':

            # determine linear noise schedule betas
            betas = torch.linspace(beta_start, beta_end, self.total_steps)

            # determine linear noise schedule alphas
            alphas = 1.0 - betas

        # case: quadratic noise scheduler
        elif scheduler == 'quad':

            # determine quadratic noise schedule betas
            betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.total_steps) ** 2

            # determine quadratic noise schedule alphas
            alphas = 1.0 - betas

        # return noise scheduler alphas and betas
        return alphas.to(self.device), betas.to(self.device)

    # define random timesteps sampler
    def sample_random_timesteps(self, n: int):

        # sample random timesteps
        t = torch.randint(low=1, high=self.total_steps, size=(n,), device=self.device)

        # return random timesteps
        return t

    # define gaussian noise addition
    def add_gauss_noise(self, x_num, t):

        # determine noise alpha hat
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None]

        # determine noise one minius alpha hat
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t])[:, None]

        # determine numeric noise
        noise_num = torch.randn_like(x_num)

        # determine x numeric noise
        x_noise_num = sqrt_alpha_hat * x_num + sqrt_one_minus_alpha_hat * noise_num

        # return x numeric noise and numeric noise
        return x_noise_num, noise_num

    # define gaussian noise sampling
    def p_sample_gauss(self, model_out, z_norm, timesteps):

        self.alphas = self.alphas.to("cuda")
        self.betas = self.betas.to("cuda")
        self.alphas_hat = self.alphas_hat.to("cuda")

        # determine noise alpha hat
        sqrt_alpha_t = torch.sqrt(self.alphas[timesteps])[:, None]

        # determine noise betas
        betas_t = self.betas[timesteps][:, None]

        # determine noise one minius alpha hat
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alphas_hat[timesteps])[:, None]

        epsilon_t = torch.sqrt(self.betas[timesteps][:, None])

        # determine random noise
        random_noise = torch.randn_like(z_norm)
        random_noise[timesteps == 0] = 0.0

        # determine model mean
        model_mean = ((1 / sqrt_alpha_t) * (z_norm - (betas_t * model_out / sqrt_one_minus_alpha_hat_t)))

        # determine z norm
        z_norm = model_mean + (epsilon_t * random_noise)

        # return z norm
        return z_norm

class FINDIFF(BaseSynthesizer):
    def __init__(self, cat_embedding_dim=2, mlp_dim=(1024, 1024, 1024, 1024), mlp_activation='lrelu',
                 diffusion_steps = 500, diffusion_beta_start = 1e-4, diffusion_beta_end = 0.02,
                 diffusion_scheduler = 'linear', mlp_lr=1e-4, epochs= 500, batch_size=512, verbose=False, cuda=True, use_wandb= False):

        self._epochs = epochs
        self._batch_size = batch_size

        self._cat_embedding_dim = cat_embedding_dim
        self._mlp_dim = mlp_dim
        self._mlp_activation = mlp_activation
        self._mlp_lr = mlp_lr
        self._diffusion_steps = diffusion_steps
        self._diffusion_beta_start = diffusion_beta_start
        self._diffusion_beta_end = diffusion_beta_end
        self._diffusion_scheduler = diffusion_scheduler
        self._verbose = verbose
        self._use_wandb = use_wandb

        self._discrete_columns = ()
        self._vocab_per_attr = {}

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

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

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):

        self._discrete_columns = discrete_columns
        self._validate_discrete_columns(train_data, discrete_columns)
        self._continous_columns = list(set(train_data.columns) - set(discrete_columns))

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        # scale numerical values
        self._num_scaler = QuantileTransformer(output_distribution='normal')
        self._num_scaler.fit(train_data[self._continous_columns])
        train_num_scaled = self._num_scaler.transform(train_data[self._continous_columns])
        train_num_torch = torch.FloatTensor(train_num_scaled)

        # scale categorical values
        if len(self._discrete_columns) > 0:
            self._data_dtypes = train_data.dtypes
            train_data[self._discrete_columns] = train_data[self._discrete_columns].astype(str)
            vocabulary_classes = np.unique(train_data[self._discrete_columns])
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(vocabulary_classes)
            train_cat_scaled = train_data[self._discrete_columns].apply(self._label_encoder.transform)
            self._vocab_per_attr = {cat_attr: set(train_cat_scaled[cat_attr]) for cat_attr in self._discrete_columns}
            train_cat_torch = torch.LongTensor(train_cat_scaled.values)

            # init tensor dataset
            train_set = TensorDataset(
                train_cat_torch,  # categorical attributes
                train_num_torch  # numerical attribute
            )

        else:
            # init tensor dataset
            train_set = TensorDataset(
                train_num_torch  # numerical attribute
            )

        dataloader = DataLoader(
            dataset=train_set,  # training dataset
            batch_size= self._batch_size,  # training batch size
            shuffle=True  # shuffle training data
        )

        if len(self._discrete_columns) > 0:
            # determine number unique categorical tokens
            n_cat_tokens = len(np.unique(train_data[self._discrete_columns]))
            # determine total categorical embedding dimension
            cat_dim = self._cat_embedding_dim * len(self._discrete_columns)
            # determine total numerical embedding dimension
            num_dim = len(set(train_data.columns) - set(self._discrete_columns))
            # determine total embedding dimension
            self._encoded_dim = cat_dim + num_dim
        else:
            n_cat_tokens = 0
            cat_dim = 0
            self._encoded_dim = len(set(train_data.columns) - set(self._discrete_columns))

        self._synthesizer_model = MLPSynthesizer(
            d_in= self._encoded_dim,
            hidden_layers= self._mlp_dim,
            activation= self._mlp_activation,
            n_cat_tokens= n_cat_tokens,
            n_cat_emb= self._cat_embedding_dim,
            embedding_learned= False
        )

        # initialize the FinDiff base diffuser model
        self._diffuser_model = BaseDiffuser(
            total_steps= self._diffusion_steps,
            beta_start= self._diffusion_beta_start,
            beta_end= self._diffusion_beta_end,
            device=self._device,
            scheduler= self._diffusion_scheduler
        )

        # determine synthesizer model parameters
        parameters = filter(lambda p: p.requires_grad, self._synthesizer_model.parameters())
        # init Adam optimizer
        optimizer = optim.Adam(parameters, lr=self._mlp_lr)
        # init learning rate scheduler
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs, verbose=False)
        # int mean-squared-error loss
        loss_fnc = nn.MSELoss()

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Loss'])

        self._synthesizer_model.train()
        self._synthesizer_model = self._synthesizer_model.to(self._device)

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Loss ({loss:.2f})'
            epoch_iterator.set_description(description.format(loss=0))

        for i in epoch_iterator:
            batch_losses = []

            if len(self._discrete_columns) > 0:
                for batch_cat, batch_num in dataloader:
                    # move tensors to device
                    batch_cat = batch_cat.to(self._device)
                    batch_num = batch_num.to(self._device)

                    # sample diffusion timestep
                    timesteps = self._diffuser_model.sample_random_timesteps(n=batch_cat.shape[0])

                    # determine categorical embeddings
                    batch_cat_emb = self._synthesizer_model.embed_categorical(x_cat=batch_cat)

                    # concatenate categorical and numerical embeddings
                    batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

                    # add diffuser gaussian noise
                    batch_noise_t, noise_t = self._diffuser_model.add_gauss_noise(x_num=batch_cat_num, t=timesteps)

                    # conduct synthesizer model forward pass
                    predicted_noise = self._synthesizer_model(x=batch_noise_t, timesteps=timesteps)

                    # compute training batch loss
                    batch_loss = loss_fnc(input=noise_t, target=predicted_noise)

                    # reset model gradients
                    optimizer.zero_grad()

                    # run model backward pass
                    batch_loss.backward()

                    # optimize model parameters
                    optimizer.step()

                    # collect training batch losses
                    batch_losses.append(batch_loss.detach().cpu().numpy())

            else:
                for batch_num in dataloader:
                    # move tensors to device
                    batch_num = batch_num[0].to(self._device)

                    # sample diffusion timestep
                    timesteps = self._diffuser_model.sample_random_timesteps(n=batch_num.shape[0])

                    # add diffuser gaussian noise
                    batch_noise_t, noise_t = self._diffuser_model.add_gauss_noise(x_num=batch_num, t=timesteps)

                    # conduct synthesizer model forward pass
                    predicted_noise = self._synthesizer_model(x=batch_noise_t, timesteps=timesteps)

                    # compute training batch loss
                    batch_loss = loss_fnc(input=noise_t, target=predicted_noise)

                    # reset model gradients
                    optimizer.zero_grad()

                    # run model backward pass
                    batch_loss.backward()

                    # optimize model parameters
                    optimizer.step()

                    # collect training batch losses
                    batch_losses.append(batch_loss.detach().cpu().numpy())

            # determine mean training epoch loss
            batch_losses_mean = np.mean(np.array(batch_losses))

            # update learning rate scheduler
            lr_scheduler.step()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Loss': [batch_losses_mean],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(loss= batch_losses_mean)
                )
            if self._use_wandb:
                wandb.log({'Loss': batch_losses_mean})

    @random_state
    def sample(self, n):
        samples = torch.randn((n, self._encoded_dim), device= self._device)

        with torch.no_grad():
            for i in reversed(range(0, self._diffusion_steps)):
                # init diffusion timesteps
                timesteps = torch.full((n,), i, dtype=torch.long, device= self._device)
                model_out = self._synthesizer_model(x= samples.float(), timesteps=timesteps).to(self._device)
                samples = self._diffuser_model.p_sample_gauss(model_out, samples, timesteps)

        # split sample into numeric and categorical parts
        samples = samples.detach().cpu().numpy()
        samples_num = samples[:, len(self._discrete_columns)*self._cat_embedding_dim:]
        samples_cat = samples[:, :len(self._discrete_columns)*self._cat_embedding_dim]

        # denormalize numeric attributes
        z_norm_upscaled = self._num_scaler.inverse_transform(samples_num)
        z_norm_df = pd.DataFrame(z_norm_upscaled, columns= self._continous_columns)

        # get embedding lookup matrix
        embedding_lookup = self._synthesizer_model.get_embeddings().cpu()

        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        samples_cat = samples_cat.reshape(-1, len(self._discrete_columns), self._cat_embedding_dim)

        # compute pairwise distances
        distances = torch.cdist(x1=embedding_lookup, x2=torch.Tensor(samples_cat))

        # get the closest distance based on the embeddings that belong to a column category
        z_cat_df = pd.DataFrame(index=range(len(samples_cat)), columns= self._discrete_columns)

        nearest_dist_df = pd.DataFrame(index=range(len(samples_cat)), columns= self._discrete_columns)

        # iterate over categorical attributes
        for attr_idx, attr_name in enumerate(self._discrete_columns):
            attr_emb_idx = list(self._vocab_per_attr[attr_name])
            attr_distances = distances[:, attr_emb_idx, attr_idx]

            nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
            nearest_idx = nearest_idx.cpu().numpy()

            z_cat_df[attr_name] = np.array(attr_emb_idx)[nearest_idx]  # need to map emb indices back to column indices
            nearest_dist_df[attr_name] = nearest_values.cpu().numpy()

        z_cat_df = z_cat_df.apply(self._label_encoder.inverse_transform)

        samples_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)

        for column, dtype in zip(self._data_dtypes.index, self._data_dtypes):
            if dtype == "bool":
                samples_decoded[column] = samples_decoded[column] == "True"
            samples_decoded[column] = samples_decoded[column].astype(dtype)

        return samples_decoded

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._synthesizer_model is not None:
            self._synthesizer_model.to(self._device)
class FINDIFFSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``FinDiff`` model.

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
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 locales=['en_US'], cat_embedding_dim=2, mlp_dim=(1024, 1024, 1024, 1024), mlp_activation='lrelu',
                 diffusion_steps = 500, diffusion_beta_start = 1e-4, diffusion_beta_end = 0.02,
                 diffusion_scheduler = 'linear', mlp_lr=1e-4, epochs= 500, batch_size=512, verbose=False, cuda=True, use_wandb= False):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        self.cat_embedding_dim = cat_embedding_dim
        self.mlp_dim = mlp_dim
        self.mlp_activation = mlp_activation
        self.diffusion_steps = diffusion_steps
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        self.diffusion_scheduler = diffusion_scheduler
        self.mlp_lr = mlp_lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_wandb = use_wandb
        self.epochs = epochs
        self.cuda = cuda

        self._model_kwargs = {
            'cat_embedding_dim': cat_embedding_dim,
            'mlp_dim': mlp_dim,
            'mlp_activation': mlp_activation,
            'diffusion_steps': diffusion_steps,
            'diffusion_beta_start': diffusion_beta_start,
            'diffusion_beta_end': diffusion_beta_end,
            'diffusion_scheduler': diffusion_scheduler,
            'batch_size': batch_size,
            'mlp_lr': mlp_lr,
            'verbose': verbose,
            'epochs': epochs,
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
            header = {'Original Column Name  ': 'Est # of Columns (CTGAN)'}
            dict_generated_columns = {**header, **dict_generated_columns}
            longest_column_name = len(max(dict_generated_columns, key=len))
            cap = '<' + str(longest_column_name)
            lines_to_print = []
            for column, num_generated_columns in dict_generated_columns.items():
                lines_to_print.append(f'{column:{cap}} {num_generated_columns}')

            generated_columns_str = '\n'.join(lines_to_print)
            print(  # noqa: T001
                'PerformanceAlert: Using the CTGANSynthesizer on this data is not recommended. '
                'To model this data, CTGAN will generate a large number of columns.'
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
        self._model = FINDIFF(**self._model_kwargs)
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

        raise NotImplementedError("CTGANSynthesizer doesn't support conditional sampling.")