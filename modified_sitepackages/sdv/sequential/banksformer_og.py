import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .transformer_core import *

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out1)

        return out3, attn_weights_block1


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, inp_dim, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.input_layer = nn.Sequential(
            nn.Linear(inp_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, look_ahead_mask, padding_mask):
        x = self.input_layer(x)

        seq_len = x.size(1)
        attention_weights = {}

        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1

        return x, attention_weights


class Transformer(nn.Module):
    def __init__(self, inp_dim, activations, num_layers_enc = None, num_layers_dec= 4, d_model= 128, num_heads= 2, dff= 128, maximum_position_encoding= 256, net_info= None, final_dim= None, rate=0.1):
        super(Transformer, self).__init__()

        self.decoder = Decoder(num_layers_dec, d_model, num_heads, dff, maximum_position_encoding, inp_dim, rate)
        self.final_layer = nn.Linear(d_model, d_model)

        assert sum(inp_dim) == len(activations)


        for name, dim in self.FIELD_DIMS_NET.items():
            acti = self.ACTIVATIONS.get(name, None)
            self.__setattr__(name, nn.Linear(dim, acti))

        self.train_loss = nn.MSELoss()
        self.results = {x: [] for x in ["loss", "val_loss", "val_loss_full", "parts"]}

    def forward(self, tar, training, look_ahead_mask, dec_padding_mask):
        tar_inp = tar[:, :-1]
        tar_out = tar[:, 1:]

        dec_output, attention_weights = self.decoder(tar_inp, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        preds = {}

        for net_name in self.ORDER:
            pred = self.__getattribute__(net_name)(final_output)
            preds[net_name] = pred

            st = self.FIELD_STARTS_IN[net_name]
            end = st + self.FIELD_DIMS_IN[net_name]
            to_add = tar_out[:, :, st:end]

            final_output = torch.cat([final_output, to_add], dim=-1)

        return preds, attention_weights

    def train_step(self, inp, tar, optimizer):
        combined_mask, dec_padding_mask = create_masks(tar)

        optimizer.zero_grad()
        predictions, _ = self(inp, True, combined_mask, dec_padding_mask)
        loss = self.loss_function(tar, predictions)

        loss.backward()
        optimizer.step()

        self.train_loss(loss.item())

    def val_step(self, inp, tar):
        combined_mask, dec_padding_mask = create_masks(tar)

        predictions, _ = self(inp, False, combined_mask, dec_padding_mask)
        return self.loss_function(tar, predictions)

    def fit(self, train_batches, x_cv, y_cv, epochs, early_stop=2, print_every=50, ckpt_every=2, mid_epoch_updates=None):
        warned_acc = False

        if mid_epoch_updates:
            batch_per_update = len(train_batches) // mid_epoch_updates

        for epoch in range(epochs):
            start = time.time()

            for batch_no, (inp, tar) in enumerate(train_batches):
                self.train_step(inp, tar)

                if batch_no % print_every == 0:
                    print(f'Epoch {epoch + 1} Batch {batch_no} Loss {self.train_loss:.4f}')

                if mid_epoch_updates and batch_no % batch_per_update == 0:
                    v_loss, *vl_parts = self.val_step(x_cv, y_cv)
                    self.results["loss"].append(self.train_loss.result().numpy())
                    self.results["val_loss"].append(v_loss)
                    self.results["parts"].append(vl_parts)

                    try:
                        acc_res = self.acc_function()
                        self.results.setdefault("val_acc", []).append(acc_res)
                    except Exception as e:
                        if not warned_acc:
                            warned_acc = True
                            print("Not recording acc:", e)

            print(f'Epoch {epoch + 1} Loss {self.train_loss:.4f}')
            v_loss, *vl_parts = self.val_step(x_cv, y_cv)
            print(f"** on validation data loss is {v_loss:.4f}")

            self.results["loss"].append(self.train_loss.result().numpy())
            self.results["val_loss"].append(v_loss)
            self.results["parts"].append(vl_parts)

            try:
                acc_res = self.acc_function()
                self.results.setdefault("val_acc", []).append(acc_res)
                print(f"** on validation data acc is \n{acc_res}")
            except Exception as e:
                if not warned_acc:
                    warned_acc = True
                    print("Not recording acc:", e)

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

            if min(self.results["val_loss"]) < min(self.results["val_loss"][-early_stop:]):
                print(f"Stopping early, last {early_stop} val losses are: {self.results['val_loss'][-early_stop:]} \nBest was {min(self.results['val_loss']):.3f}\n\n")
                break

            if (epoch + 1) % ckpt_every == 0:
                torch.save(self.state_dict(), f'checkpoint_epoch_{epoch + 1}.pth')
                print(f'Saving checkpoint for epoch {epoch + 1} at checkpoint_epoch_{epoch + 1}.pth')
