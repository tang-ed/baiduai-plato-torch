import jieba
import json
from collections import OrderedDict
import numpy as np
import re


class Tokenizer:
    def __init__(self, is_jieba=False, ignore=["\n", "\t", "\r", " ", ""], token_ids=["[null]", "[start]", "[end]", "[sep]", "[unk]"]):
        self.ignore = ignore
        self.token_ids = token_ids
        self.is_jieba = is_jieba
        self.word_index = {v:k for k, v in enumerate(token_ids)}
        self.max_len = None
        self.word_dict = OrderedDict()

    def jieba_split(self, sentence):
        jieba_data = jieba.cut(sentence)
        return list(jieba_data)

    def add_word(self, word):
        if word not in self.ignore:
            if word not in self.word_dict:
                self.word_dict[word] = 1
            else:
                self.word_dict[word] += 1

    def add_word_num(self, s):

        if s not in self.ignore:
            try:
                num = self.word_index[s]
            except:
                num = self.word_index["[unk]"]
            return num
        else:
            return None

    def fit_text(self, text, save_wd_idx=True, split_space=False):
        """
        生成 word_index
        :param text: [sentence1, sentence2 ....]
        :return:
        """
        if not split_space:
            self.max_len = max([len(i) for i in text]) + 4
        else:
            self.max_len = max([len(i.split(" ")) for i in text]) + 4
        for sentence in text:
            if not self.is_jieba:
                if split_space:
                    for word in sentence.split(" "):
                        self.add_word(word)
                else:
                    for word in sentence:
                        self.add_word(word)
            else:
                for word in self.jieba_split(sentence):
                    self.add_word(word)

        word_index = list(self.word_dict.items())
        word_index.sort(key=lambda x:x[1], reverse=True)

        for i in word_index:
            word = i[0]
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index)

        self.word_index["max_len"] = str(self.max_len)

        if save_wd_idx:
            json_data = json.dumps(self.word_index, indent=4, ensure_ascii=False)
            with open("word_index.json", "w", encoding="utf-8") as f:
                f.write(json_data)

    @classmethod
    def from_json_file(cls, file):
        token = Tokenizer()
        with open(file, "r", encoding="utf-8") as f:
            token.word_index = json.load(f)

        return token

    def add_sentence_token(self, sentence, add_token, split_space):
        nums = []
        if not self.is_jieba:
            if split_space:
                for s in sentence.split(" "):
                    num = self.add_word_num(s)
                    if num is None:
                        continue
                    nums.append(num)
            else:
                for s in sentence:
                    num = self.add_word_num(s)
                    if num is None:
                        continue
                    nums.append(num)
        else:
            for s in self.jieba_split(sentence):
                if s not in self.ignore:
                    try:
                        num = self.word_index[s]
                    except:
                        num = self.word_index["[unk]"]
                    nums.append(num)
        if add_token:
            nums.append(self.word_index["[end]"])
            nums.insert(0, self.word_index["[start]"])
        return nums

    def encoder_sentence(self, sentences, add_token=True, split_space=True):
        """
        对sentence进行编码
        :param sentences: [sentence1, sentence2 ....]
        :param add_token: 是否添加开头和结尾标记
        :return: [sentence_num1, sentence_num2 ....]
        """

        if type(sentences) is list:
            new_sentence = []
            for sentence in sentences:
                nums = self.add_sentence_token(sentence, add_token, split_space)
                new_sentence.append(nums)
            return new_sentence
        else:
            return self.add_sentence_token(sentences, add_token, split_space)

    def padding_sentence(self, sentence, padding, max_len, mask, idx=0):
        sentence_len = len(sentence)

        if padding == "post":
            if max_len+2 < self.max_len:
                mask[idx, :sentence_len] = sentence
            else:
                mask[idx, :-2] = sentence[-(self.max_len-2):]
                mask[idx, 0] = 1
        else:
            if max_len+2 < self.max_len:
                mask[idx, -sentence_len:] = sentence
            else:
                mask[idx, :-2] = sentence[-(max_len-2):]
        return mask

    def padding(self, sentences, padding="post", pad_num=0, max_len=-1):
        """

        :param sentences: [sentence_num1, sentence_num2 ....]
        :param padding: "post"为前填充，非即为后填充
        :param pad_num: 填充的标记
        :param max_len: 限定长度，默认为最长+2
        :return: arr
        """
        if max_len == -1:
            max_len = max([len(i) for i in sentences]) + 2

            if max_len > self.max_len:
                max_len = self.max_len


        if type(sentences[0]) is list:
            mask = np.zeros((len(sentences), max_len), dtype="int64")
            for idx, sentence in enumerate(sentences):
                mask = self.padding_sentence(sentence, padding, max_len, mask, idx)
        else:
            mask = np.zeros((1, max_len), dtype="int64")
            mask = self.padding_sentence(sentences, padding, max_len, mask)[0]
        return mask

    def return_max_len(self, sentences, split_space=False):
        if split_space:
            max_len = max([len(i.split(" ")) for i in sentences])
        else:
            max_len = max([len(i) for i in sentences])
        self.max_len = max_len+4
        return max_len + 4

    def return_max_sentence(self, sentence1, sentence2=None, split_space=False):
        if sentence2 is None:
            sentence2 = sentence1
            sentence_one = True
        else:
            sentence_one = False
        len1 = len(sentence1)
        len2 = len(sentence2)

        new_sentence1 = []
        new_sentence2 = []

        if len1 != len2:
            assert "len1 必须等于 len2"
        for i in range(len(sentence1)):
            if split_space:
                if len(sentence1[i].split(" ")) <= self.max_len-4:
                    new_sentence1.append(sentence1[i])
                    new_sentence2.append(sentence2[i])
            else:
                if len(sentence1[i]) <= self.max_len-4:
                    new_sentence1.append(sentence1[i])
                    new_sentence2.append(sentence2[i])
        if not sentence_one:
            return new_sentence1, new_sentence2
        else:
            return new_sentence1

    def decoder_num(self, sentence_num, save_token=True):
        word_index = {v:k for k, v in self.word_index.items()}

        sentences= []
        if len(sentence_num.shape) == 2:
            for sentence in sentence_num:
                sentence_ls = []
                for num in sentence:
                    word = word_index[num]
                    if not save_token:
                        if word in self.token_ids:
                            continue

                    sentence_ls.append(word)
                sentences.append(sentence_ls)
        else:
            for num in list(sentence_num.numpy()):

                word = word_index[num]
                if not save_token:
                    if word in self.token_ids:
                        continue
                sentences.append(word)
        return sentences

if __name__ == '__main__':
    text = ["哈哈哈哈！", "怎么拉"]
    token = Tokenizer(is_jieba=True)
    token.fit_text(text, save_wd_idx=True)
    st = token.encoder_sentence(text)
    data = token.padding(st)
    print(data)



