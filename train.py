import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tokenizer import Tokenizer
from UnifiedTransformer import UnifiedTransformerModel, UnifiedTransformerConfig
from tqdm import tqdm
import os
import numpy as np

# 参数定义
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 24
lr = 1e-4
epochs = 50
config = UnifiedTransformerConfig.from_json_file("config.json")
tokenizer = None


def expand_attention_mask(attention_mask, tgt_len=None):
    mask = attention_mask[:, None, None, :]

    baz, seq_len = attention_mask.shape
    tgt_len = tgt_len if tgt_len is not None else seq_len

    mask = 1.0 - mask

    mask = mask.expand(1, 1, tgt_len, seq_len)

    return mask.masked_fill(mask.bool(), -10000000.0)

def make_attention_causal(shape, causal_index, dtype=torch.float32, att_mask=None):
    baz, seq_len = shape

    causal = None
    if seq_len > 1:
        causal = torch.full((seq_len, seq_len), fill_value=-10000000.0, device=device, dtype=dtype)

        causal.masked_fill_(causal_index < (causal_index + 1).view(-1, 1), 0)

        causal = causal[None, None, :, :].expand(1, 1, seq_len, seq_len)

        if att_mask is not None:
            att_mask = expand_attention_mask(att_mask)
            causal = att_mask + causal

    return causal



def read_data():

    with open("sentence_data.txt", "r", encoding="utf-8") as f:
        datas = f.readlines()

    return datas

def create_data(datas):
    global tokenizer
    values = []
    labels = []

    i_s = ""
    for i, n in zip(datas[:-1], datas[1:]):
        if i == "\n" or n == "\n":
            i_s = ""
            continue
        if i_s == "":
            i_s = i
        else:
            i_s = i_s + " " + i

        values.append(i_s)
        labels.append(n)
    if not os.path.exists("word_index.json"):
        tokenizer = Tokenizer()
        tokenizer.fit_text(values + labels, split_space=False)
    else:
        tokenizer = Tokenizer.from_json_file("word_index.json")

    tokenizer.max_len = config.max_position_embeddings

    return values, labels

class ChatData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def pre_data(self, x):
        words = x.split(" ")
        new_words = []
        for w in words:
            if w == "[sep]":
                new_words.append(w)
            elif w == "":
                continue
            else:
                for i in w:
                    new_words.append(i)


        return " ".join(new_words[-(config.max_position_embeddings-10):])

    def __getitem__(self, item):
        value, label = self.values[item], self.labels[item]

        values = value.split("\n")[:-1]

        values = " [sep] ".join(values)

        label = label[:-1]
        value = self.pre_data(values)
        label = self.pre_data(label)

        value = tokenizer.encoder_sentence(value, add_token=False)
        label = tokenizer.encoder_sentence(label, add_token=True)

        mask = np.zeros((len(value),), dtype="int32")
        type_token_ids = list(np.concatenate([mask, np.ones((len(label),), dtype="int32")]))

        attention_range = list(np.concatenate([mask, np.arange(1, len(label)+1, dtype="int32")]))

        value = value + label
        value = value[-(config.max_position_embeddings-10):]
        type_token_ids = type_token_ids[-(config.max_position_embeddings-10):]
        attention_range = attention_range[-(config.max_position_embeddings-10):]

        return value, label[1:]+[0], type_token_ids, attention_range

    def __len__(self):
        return len(self.values)

    @staticmethod
    def call_fc(batch):

        xs  = []
        ts  = []
        ys  = []
        ats = []
        label_point = []
        attention_mask = []
        for x, y, t, a in batch:
            xs.append(x)
            ts.append(t)
            ys.extend(y)
            ats.append(a)

        input_ids = tokenizer.padding(xs, padding="no_post")
        type_token_ids = tokenizer.padding(ts, padding="no_post")
        attention_range = tokenizer.padding(ats, padding="no_post")

        label_ids = np.array(ys)
        indexes = np.where(type_token_ids == 1)
        lens = input_ids.shape[-1]
        for xz, yz in zip(indexes[0], indexes[1]):
            label_point.append(xz*lens+yz)
        label_ids = torch.LongTensor(label_ids).to(device)
        input_ids = torch.IntTensor(input_ids).to(device)
        type_token_ids = torch.IntTensor(type_token_ids).to(device)
        attention_range = torch.IntTensor(attention_range).to(device)
        label_point = torch.LongTensor(label_point).to(device)

        shape = input_ids.shape
        for i, a in zip(input_ids, attention_range):
            att = (i != 0).int()[None, :]
            attc = make_attention_causal(shape, a, att_mask=att)
            attention_mask.append(attc)
        attention_mask = torch.cat(attention_mask, dim=0)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "type_token_ids": type_token_ids,
            "masked_positions": label_point,
        }

        return inputs, label_ids

def train():
    datas = read_data()
    values, labels = create_data(datas)

    train_data = ChatData(values, labels)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=train_data.call_fc)

    model = UnifiedTransformerModel(config)
    # model.load_state_dict(torch.load("model.pkl"))
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss(ignore_index=0)

    old_loss = 100

    train_ls = []

    for epoch in range(1, epochs+1):
        pbar = tqdm(train_data)
        loss_all = 0
        all_acc = 0

        for step, (x, y) in enumerate(pbar):

            out = model(**x)

            loss = loss_fc(out.reshape((-1, out.shape[-1])), y.reshape((-1,)))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_all += loss.item()
            loss_time = loss_all / (step + 1)

            y_mask = (y > 0).int()
            word_all = torch.sum(y_mask)
            y_mask[y_mask == 0] = -1

            out_y = torch.argmax(out, dim=-1)

            out_y = out_y * y_mask
            acc = torch.sum((out_y == y).float()) / word_all

            all_acc += acc
            acc_time = all_acc / (step + 1)

            s = (
                "train => epoch: {} - step: {} - loss: {:.3f} - loss_time: {:.3f} - acc: {:.3f} - acc_time:{:.3f}".format(
                    epoch, step, loss, loss_time, acc, acc_time))

            train_ls.append(s+"\n")

            pbar.set_description(s)

        if old_loss > loss_time:
            old_loss = loss_time
            torch.save(model.state_dict(), "model.pkl")

        with open("train_result.txt", "w") as f:
            f.writelines(train_ls)



if __name__ == '__main__':
    train()