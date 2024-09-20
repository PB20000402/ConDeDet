import argparse
import torch
import random
import numpy as np
import logging
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pandas as pd

from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
import os
import json


def set_seed(args):
    # 设置 Python 的随机数生成器的种子
    random.seed(args.seed)

    # 设置 NumPy 库的随机数生成器的种子
    np.random.seed(args.seed)

    # 设置 PyTorch 库的随机数生成器的种子
    torch.manual_seed(args.seed)

    # 如果有可用的 GPU，则设置所有 CUDA 设备的随机数生成器的种子
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别，记录所有消息 colab中加这个才会有日志信息 pycharm中不需要加这个
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        # 全连接层，输入维度为config.hidden_size，输出维度也为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout 层，防止过拟合，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 输出层，将特征映射到两个类别（二分类任务）
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        # 取第一个token（[CLS]标记）的特征
        x = features[:, 0, :]
        x = self.dropout(x)  # 使用Dropout进行正则化
        x = self.dense(x)  # 全连接层
        x = torch.tanh(x)  # 使用tanh激活函数
        x = self.dropout(x)  # 使用Dropout进行正则化
        x = self.out_proj(x)  # 输出层
        return x

class Model(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob


# 定义一个空函数 warn 用于禁用警告输出
def warn(*args, **kwargs):
    pass


# 将 warnings.warn 函数指向 warn 函数，实现禁用警告输出
import warnings

warnings.warn = warn


# 定义一个用于表示单个训练/测试样本的类 InputFeatures
class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 #idx,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        #self.idx = str(idx)
        self.label = label


# 将文本转化为模型可以理解的特征
def convert_examples_to_features(js, tokenizer, args):
    # 从 JSON 中提取代码并进行分词
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]

    # 构建输入 tokens 和对应的 token IDs
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    # 进行填充，保证长度为 args.block_size
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    #return InputFeatures(source_tokens, source_ids, js['idx'], js['target'])
    return InputFeatures(source_tokens, source_ids, js['target'])


# 清空 GPU 缓存
torch.cuda.empty_cache()


# 定义一个数据集类 TextDataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        # 如果指定了采样比例，对样本进行随机采样
        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        # 输出一些日志信息
        if 'train' in file_path:
            logger.info("*** Total Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Sample ***")
                logger.info("Total sample".format(idx))
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)  # 使用随机采样
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)  # 创建数据加载器

    # 设定一些训练的参数
    args.max_steps = args.epochs * len(train_dataloader)  # 最大步数
    args.save_steps = len(train_dataloader)  # 保存模型的间隔步数
    args.warmup_steps = args.max_steps // 5  # 预热步数
    model.to(args.device)  # 将模型移至指定设备

    # 准备优化器和学习率调度器（线性预热和衰减）
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # 多GPU训练
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 训练开始
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0  # 全局训练步数
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0  # 初始化损失值等
    best_f1 = 0  # 记录最好的 F1 值

    model.zero_grad()  # 梯度清零

    for idx in range(args.epochs):  # 遍历每个 epoch
        torch.cuda.empty_cache()  # 每个epoch开始释放无用的显存
        bar = tqdm(train_dataloader,total=len(train_dataloader))  # 创建进度条
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):  # 遍历每个 batch
            (inputs_ids, labels) = [x.to(args.device) for x in batch]  # 将输入数据移至指定设备
            model.train()  # 设置模型为训练模式
            loss, logits = model(input_ids=inputs_ids, labels=labels)  # 前向传播计算损失
            if args.n_gpu > 1:
                loss = loss.mean()  # 如果使用多GPU，需要对损失进行均值计算

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps  # 如果使用梯度累积，需要对损失进行缩放

            loss.backward()  # 反向传播计算梯度
            torch.cuda.empty_cache()  # backward之后 optimizer之前
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 对梯度进行裁剪

            tr_loss += loss.item()  # 累加损失
            tr_num += 1
            train_loss += loss.item()  # 累加训练损失
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num, 5)  # 计算平均损失
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))  # 更新进度条的描述

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 梯度清零
                scheduler.step()  # 更新学习率
                torch.cuda.empty_cache()  # 添加在optimizer.step()之后
                global_step += 1  # 更新全局训练步数
                output_flag = True  # 输出标志
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)), 4)  # 计算平均损失

                if global_step % args.save_steps == 0:  # 如果满足保存间隔
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)  # 在训练过程中评估模型
                    torch.cuda.empty_cache()  # 添加在evaluate之后

                    # 保存模型
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']  # 更新最好的 F1 值
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        #checkpoint_prefix = 'checkpoint-best-f1-comment'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)  # 保存模型权重
                        logger.info("Saving model checkpoint to %s", output_dir)  # 输出模型保存路径
    torch.cuda.empty_cache()  # 清空GPU缓存

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # 创建用于评估的数据加载器
    eval_sampler = SequentialSampler(eval_dataset)  # 使用顺序采样
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size)

    # 多GPU评估
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # 进行评估
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0  # 初始化评估损失
    nb_eval_steps = 0  # 初始化评估步数
    model.eval()  # 设置模型为评估模式
    logits=[]
    y_trues=[]
    for batch in eval_dataloader:  # 遍历每个 batch
        (inputs_ids, labels)=[x.to(args.device) for x in batch]  # 将输入数据移至指定设备
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)  # 前向传播计算损失和预测
            eval_loss += lm_loss.mean().item()  # 累加评估损失
            logits.append(logit.cpu().numpy())  # 将预测结果存储
            y_trues.append(labels.cpu().numpy())  # 将真实标签存储
        nb_eval_steps += 1  # 更新评估步数
        torch.cuda.empty_cache()  # 添加每轮结束后释放显存

    # 计算评估指标
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5  # 阈值设定为0.5
    best_f1 = 0  # 初始化最好的 F1 值
    y_preds = logits[:, 1] > best_threshold  # 根据预测概率和阈值得到预测类别
    recall = recall_score(y_trues, y_preds)  # 计算召回率
    precision = precision_score(y_trues, y_preds)  # 计算精确率
    f1 = f1_score(y_trues, y_preds)  # 计算 F1 值
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }

    # 输出评估结果
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    torch.cuda.empty_cache() # 预测结束后释放显存
    return result

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # 创建用于测试的数据加载器
    test_sampler = SequentialSampler(test_dataset)  # 使用顺序采样
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # 多GPU测试
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 进行测试
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0  # 初始化测试损失
    nb_eval_steps = 0  # 初始化测试步数
    model.eval()  # 设置模型为评估模式
    logits=[]
    y_trues=[]
    for batch in test_dataloader:  # 遍历每个 batch
        (inputs_ids, labels) = [x.to(args.device) for x in batch]  # 将输入数据移至指定设备
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)  # 前向传播计算损失和预测
            eval_loss += lm_loss.mean().item()  # 累加测试损失
            logits.append(logit.cpu().numpy())  # 将预测结果存储
            y_trues.append(labels.cpu().numpy())  # 将真实标签存储
        nb_eval_steps += 1  # 更新测试步数
        torch.cuda.empty_cache()  # 添加每轮结束后释放显存

    # 计算测试指标
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold  # 根据预测概率和阈值得到预测类别
    acc = accuracy_score(y_trues, y_preds)  # 计算准确率
    recall = recall_score(y_trues, y_preds)  # 计算召回率
    precision = precision_score(y_trues, y_preds)  # 计算精确率
    f1 = f1_score(y_trues, y_preds)  # 计算 F1 值
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold":best_threshold,
    }

    # 输出测试结果
    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    torch.cuda.empty_cache() # 预测结束后释放显存
    return result

def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file .")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")

    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    #args = parser.parse_args()
    args = parser.parse_args(args=[])#colab中这样用 不然会报错


    args.output_dir='e1/saved_models'
    args.model_type="roberta"
    args.tokenizer_name='model'
    args.model_name_or_path='model'
    args.do_train =False
    args.do_test =True
    args.train_data_file='e1/CodeXGLUE/train.jsonl'
#    args.train_data_file = 'e1/CodeXGLUE/comment-data/train-comment.jsonl'
    args.eval_data_file='e1/CodeXGLUE/valid.jsonl'
#    args.eval_data_file = 'e1/CodeXGLUE/comment-data/valid-comment.jsonl'
#    args.test_data_file='e1/CodeXGLUE/test.jsonl'
    args.test_data_file = 'e1/CodeXGLUE/comment-data/test-comment.jsonl'
    args.epochs= 10
    args.block_size =512
    args.train_batch_size=16
    args.eval_batch_size =16
    args.learning_rate =2e-5
    args.max_grad_norm =1.0
    args.evaluate_during_training
  # --seed 123456  2>&1 | tee train.log

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)

    # Set seed set_seed()方法 固定随机数种子
    set_seed(args)

    # 预训练模型的config 分词和模型部分下载
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('/e1/word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="/e1/bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="/e1/bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)

    #开始训练与测试
    logger.info("Training/evaluation parameters %s", args)
    torch.cuda.empty_cache()
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1-comment/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, args.test_data_file)
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()