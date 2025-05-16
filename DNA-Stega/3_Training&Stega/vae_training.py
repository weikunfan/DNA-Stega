import torch
from torch import nn
import torch.optim as optim
import scipy.stats
import numpy as np
from logger import Logger
import utils_vae  # 使用修改后的 utils_vae 模块
import vae_model  # 导入 VAE 模型
import os
import inspect
from transformers import AutoTokenizer
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def continue_training(path):
    File = []

    for root, dirs, files in os.walk(path):
        for file in files:
            File.append(os.path.join(root, file))

    Models = []
    for f in File:
        if f.find('pkl') > 0:
            Models.append(f)

    epoch_nums = []
    for m in Models:
        m_ = m[m.rfind('/') + 1: -4]
        m_name = m_.split('-')
        try:
            epoch_nums.append(int(m_name[-1]))
        except ValueError:
            continue

    try:
        return_num = str(max(epoch_nums))
    except:
        return_num = 0
    return return_num

def main(file, code, SeqLength, ind, Path_model_save, file_name, modelnum, lr, running_mode,decoder='lstm'):
    print(torch.__version__)
    print(torch.cuda.is_available())

    # ======================
    # 超参数
    # ======================
    DATASET = 'base1'
    RATIO = 0.9
    MIN_LEN = SeqLength
    MAX_LEN = 200
    BATCH_SIZE = 30
    
    # 根据解码器类型调整超参数
    if decoder == 'transformer':
        EMBED_SIZE = 512      # transformer通常使用更大的维度
        HIDDEN_DIM = 512      # transformer的d_model
        NUM_LAYERS = 6        # transformer通常使用6层
        DROPOUT_RATE = 0.1    # transformer推荐的dropout率
        N_HEADS = 8          # 注意力头数
        DIM_FEEDFORWARD = 2048  # 前馈网络维度
        WARMUP_STEPS = 4000   # 预热步数
    else:  # lstm or gru
        EMBED_SIZE = 350
        HIDDEN_DIM = 512
        NUM_LAYERS = 2
        DROPOUT_RATE = 0.2
        N_HEADS = None
        DIM_FEEDFORWARD = None
        WARMUP_STEPS = None
    if running_mode == 't':  # test
        EPOCH = 3
    elif running_mode == 's':  # start_train
        EPOCH = 50
    elif running_mode == 'c':  # continue_train
        EPOCH = 20
    LEARNING_RATE = lr
    MAX_GENERATE_LENGTH = SeqLength
    GENERATE_EVERY = 5
    PRINT_EVERY = 1
    SEED = 100
    repeat = 1
    early_stop = 0
    STOP = 100
    N_Z = 128  # 潜在向量的维度

    # 初始化 tokenizer
        # 根据 ind 的值选择不同的预训练模型路径
    if ind == '3':
        model_path = '/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_3mer'
    elif ind == '4':
        model_path = '/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_4mer'
    elif ind == '5':
        model_path = '/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_5mer'
    elif ind == '6':
        model_path = '/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_6mer'
    else:
        raise ValueError(f"Invalid ind value: {ind}. Expected '3', '4', '5', or '6'.")

    print("\n" + "="*50)
    print(f"Loading DNABERT-{ind}mer tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # tokenizer = AutoTokenizer.from_pretrained('/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_3mer')
    # 添加特殊标记
    # special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]'}
    # tokenizer.add_special_tokens(special_tokens_dict)

# 设置 bos_token 和 eos_token，与 stega_vae.py 保持一致
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.eos_token_id = tokenizer.sep_token_id
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.bos_token_id = tokenizer.cls_token_id
    vocab_size = len(tokenizer)
    vocab_size = len(tokenizer)
    # 打印特殊标记
    print("Tokenizer special tokens:")
    print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("sep_token:", tokenizer.sep_token, tokenizer.sep_token_id)
    print("cls_token:", tokenizer.cls_token, tokenizer.cls_token_id)
    all_var = locals()
    print()

    log_file = Path_model_save + r"/{}/read_{}/M_{}".format(file_name, ind, str(modelnum)) + '/{}.txt'.format(code)

    logger = Logger(log_file)
    logger.info(
        'Epoch {} Lr {} BatchSize{} EmbedSize{} HiddenDim{}'.format(EPOCH, LEARNING_RATE, BATCH_SIZE, EMBED_SIZE,
                                                                    HIDDEN_DIM))
    # ======================
    # 数据
    # ======================
    data_path = file
    train_path = Path_model_save + r"/{}/read_{}/M_{}".format(file_name, ind, str(modelnum)) + '/train.txt'
    test_path = Path_model_save + r"/{}/read_{}/M_{}".format(file_name, ind, str(modelnum)) + '/test.txt'

    # 划分数据集
    utils_vae.split_corpus(data_path, train_path, test_path, max_len=MAX_LEN, min_len=MIN_LEN, ratio=RATIO,
                           seed=SEED)
    # 构建语料库
    train = utils_vae.Corpus(train_path, tokenizer, max_len=MAX_LEN, min_len=MIN_LEN)
    test = utils_vae.Corpus(test_path, tokenizer, max_len=MAX_LEN, min_len=MIN_LEN)
    train_generator = utils_vae.Generator(train.corpus, tokenizer=tokenizer)
    test_generator = utils_vae.Generator(test.corpus, tokenizer=tokenizer)

    # ======================
    # 构建模型
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("\n" + "="*50)
    print("Initializing VAE model with:")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Embedding size: {EMBED_SIZE}")
    print(f"- Hidden dim: {HIDDEN_DIM}")
    print(f"- Number of layers: {NUM_LAYERS}")
    print(f"- Dropout rate: {DROPOUT_RATE}")
    print(f"- Latent dimension (N_Z): {N_Z}")
    
    model = vae_model.VAE(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        n_z=N_Z,
        load_pretrained=True,
        mer_value=ind,
        decoder_type=decoder,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD
    )

    print("\n" + "="*50)
    print("Model architecture:")
    print(f"- Encoder: DNABERT-{ind}mer")
    print(f"- Decoder: {decoder} with {NUM_LAYERS} layers")
    print("="*50 + "\n")

    # 更新模型的嵌入层大小以适应新的标记
    model.encoder.resize_token_embeddings(vocab_size)
    model.embedding = nn.Embedding(vocab_size, EMBED_SIZE)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {:d}".format(total_trainable_params))
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    
    # 根据解码器类型选择优化器和学习率调度器
    if decoder == 'transformer':
        # 使用Adam优化器，带有warmup的学习率调度
        optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        
        # 自定义学习率调度函数
        def transformer_lr_lambda(step):
            # 实现transformer的学习率调度策略
            step = max(1, step)  # 避免除零
            factor = min(step ** (-0.5), step * WARMUP_STEPS ** (-1.5))
            return LEARNING_RATE * factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=transformer_lr_lambda)
    else:
        # 原有的优化器和学习率调度策略
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=0.00001
        )
    print()

    # 初始化PGD
    pgd = utils_vae.PGD(model, 
              epsilon=1.0,  # 总扰动大小限制
              alpha=0.3,    # 每步扰动大小
              steps=3)      # 攻击步数
    
    # ======================
    # 训练与测试
    # ======================
    best_loss = 1e6
    step = 0
    print()
    logger.info("------------------------")

    if running_mode == 'c':
        file_load = Path_model_save + r"/{}/read_{}/M_{}".format(file_name, ind, str(modelnum))
        LOAD_EPOCH = continue_training(file_load)
    else:
        LOAD_EPOCH = 0

    if LOAD_EPOCH != 0:
        model_load = file_load + '/{}-{}-{}.pkl'.format(code, modelnum, LOAD_EPOCH)
        model.load_state_dict(torch.load(model_load, map_location=device))  # 加载模型

    for epoch in range(int(LOAD_EPOCH) + 1, EPOCH + int(LOAD_EPOCH)):
        train_g = train_generator.build_generator(BATCH_SIZE)
        test_g = test_generator.build_generator(BATCH_SIZE)
        train_loss = []
        model.train()
        while True:
            try:
                batch_inputs = next(train_g)
            except StopIteration:
                break

            # 1. 正常训练步骤
            optimizer.zero_grad()
            input_ids = batch_inputs['input_ids'].to(device)
            attention_mask = batch_inputs['attention_mask'].to(device)
            text_in = input_ids[:, :-1]
            text_target = input_ids[:, 1:]
            attention_mask_in = attention_mask[:, :-1]
            x_dec = text_in

            output, mu, logvar = model(input_ids=text_in, 
                                     attention_mask=attention_mask_in, 
                                     x_dec=x_dec)
            
            # 计算原始损失
            loss_recon = criterion(output.view(-1, vocab_size), text_target.reshape(-1))
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld = kld / input_ids.size(0)
            
            # 对于transformer，可以调整KL散度的权重
            kl_weight = 0.1 if decoder != 'transformer' else 0.2
            loss = loss_recon + kld * kl_weight

            # 反向传播和优化
            loss.backward()
            
            # 对于transformer，可以添加梯度裁剪
            if decoder == 'transformer':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 对于transformer，在每个步骤更新学习率
            if decoder == 'transformer':
                scheduler.step()

            # 2. PGD对抗训练
            pgd.attack(is_first_attack=True)  # 第一步
            for _ in range(pgd.steps - 1):
                optimizer.zero_grad()
                
                # 对抗样本的前向传播
                adv_output, adv_mu, adv_logvar = model(input_ids=text_in, 
                                                      attention_mask=attention_mask_in, 
                                                      x_dec=x_dec)
                
                # 计算对抗损失
                adv_loss_recon = criterion(adv_output.view(-1, vocab_size), 
                                         text_target.reshape(-1))
                adv_kld = -0.5 * torch.sum(1 + adv_logvar - adv_mu.pow(2) - adv_logvar.exp())
                adv_kld = adv_kld / input_ids.size(0)
                adv_loss = adv_loss_recon + adv_kld * 0.1
                
                adv_loss.backward()
                pgd.attack()  # 后续步骤
            
            # 恢复参数
            pgd.restore()
            
            # 更新参数
            optimizer.step()
            
            # 记录总损失
            train_loss.append(loss.item())
            step += 1
            torch.cuda.empty_cache()

        test_loss = []
        model.eval()
        with torch.no_grad():
            while True:
                try:
                    batch_inputs = next(test_g)
                except StopIteration:
                    break
                input_ids = batch_inputs['input_ids'].to(device)
                attention_mask = batch_inputs['attention_mask'].to(device)
                text_in = input_ids[:, :-1]
                text_target = input_ids[:, 1:]
                attention_mask_in = attention_mask[:, :-1]
                x_dec = text_in

                output, mu, logvar = model(input_ids=text_in, attention_mask=attention_mask_in, x_dec=x_dec)
                loss = utils_vae.loss_function(output, text_target, mu, logvar, criterion)
                test_loss.append(loss.item())

        # 在每个epoch结束后更新学习率
        avg_test_loss = np.mean(test_loss)
        if decoder != 'transformer':  # 只有非transformer才使用ReduceLROnPlateau
            scheduler.step(avg_test_loss)

        logger.info('epoch {:d}   training loss {:.4f}    test loss {:.4f}'
                    .format(epoch, np.mean(train_loss), np.mean(test_loss)))

        if np.mean(test_loss) < best_loss:
            best_loss = np.mean(test_loss)
            best_epoch = epoch  # 更新最佳 epoch
            print('-----------------------------------------------------')
            print('saving parameters')
            os.makedirs('models', exist_ok=True)

        # 格式化best_loss，保留小数点后四位
            formatted_loss = f"{best_loss:.4f}"
            
            # 更新保存路径以包含解码器类型
            if decoder == 'transformer':
                model_save_path = os.path.join(
                    Path_model_save, 
                    f"{file_name}/read_{ind}/transformerM_{modelnum}",
                    'lr_' + str(lr)
                )
            else:
                model_save_path = os.path.join(
                    Path_model_save, 
                    f"{file_name}/read_{ind}/full{decoder}M_{modelnum}",
                    'lr_' + str(lr)
                )

            os.makedirs(model_save_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, f'{code}-{modelnum}-{epoch}-loss_{formatted_loss}.pkl')
            )
            print('-----------------------------------------------------')
        else:
            early_stop += 1

        if early_stop >= STOP:
            return best_epoch, best_loss

        if (epoch + 1) % GENERATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                # 生成文本，���用 tokenizer.bos_token_id 或 tokenizer.cls_token_id
                x_dec = torch.LongTensor([[tokenizer.bos_token_id]] * repeat).to(device)
                generated_sequences = []
                for _ in range(repeat):
                    z = torch.randn(1, N_Z).to(device)
                    x_generated = x_dec.clone()
                    for _ in range(MAX_GENERATE_LENGTH):
                        z_expanded = z.unsqueeze(1).expand(-1, x_generated.size(1), -1)
                        x_embed_dec = model.embedding(x_generated)
                        dec_input = torch.cat([x_embed_dec, z_expanded], dim=2)
                        if model.decoder_type == 'transformer':
                            # 准备transformer解码器输入
                            dec_input = model.decoder_input_proj(dec_input)
                            dec_input = model.pos_encoder(dec_input)
                            
                            # 创建memory
                            memory = z.unsqueeze(1)  # [batch_size, 1, n_z]
                            zeros = torch.zeros(z.size(0), 1, model.hidden_dim - z.size(1)).to(z.device)
                            memory = torch.cat([memory, zeros], dim=2)  # [batch_size, 1, hidden_dim]
                            
                            # 创建tgt_mask
                            tgt_mask = model.generate_square_subsequent_mask(x_generated.size(1)).to(z.device)
                            
                            # 使用transformer解码器
                            h_dec = model.decoder(
                                dec_input,
                                memory=memory,
                                tgt_mask=tgt_mask
                            )
                        else:
                            # LSTM/GRU解码器
                            if model.decoder_type == 'lstm':
                                h_dec, _ = model.decoder(dec_input)
                            else:  # GRU
                                h_dec, _ = model.decoder(dec_input)
                        logits_output = model.output_layer(h_dec)
                        log_prob = model.log_softmax(logits_output)
                        prob = torch.exp(log_prob)[:, -1, :]
                        prob[:, tokenizer.pad_token_id] = 0  # 将 PAD 的概率设为 0
                        prob = prob / prob.sum(dim=1, keepdim=True)
                        next_word = torch.multinomial(prob, 1)
                        x_generated = torch.cat([x_generated, next_word], dim=1)
                        if next_word.item() == tokenizer.eos_token_id:
                            break
                    generated_sequences.append(x_generated.squeeze(0).cpu().numpy())

                print('-----------------------------------------------------')
                for seq in generated_sequences:
                    generated_text = tokenizer.decode(seq, skip_special_tokens=True)
                    print(generated_text)
                print('-----------------------------------------------------')

    logger.info("{}            {}".format(file, code))
    return best_epoch, best_loss
# if __name__ == '__main__':
#     # 在这里填写参数进行调用
#     # main(file, code, SeqLength, ind, Path_model_save, mode, modelnum, lr, running_mode)
#     pass
