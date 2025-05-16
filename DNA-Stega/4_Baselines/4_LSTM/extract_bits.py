import torch
import utils
import lm  # 您的语言模型模块
import os

def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index
    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up
    return index


def main_extract_bits(mode, Ranges, index, file_, Path_save):
    # ======================
    # Hyperparameters
    # ======================
    CELL = "lstm"
    EMBED_SIZE = 350
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    MAX_GENERATE_LENGTH = 200
    torch.manual_seed(42)

    # Load vocabulary
    vocabulary = utils.Vocabulary(
        file_,
        max_len=MAX_GENERATE_LENGTH,
        min_len=1,
        word_drop=0
    )

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lm.LM(
        cell=CELL,
        vocab_size=vocabulary.vocab_size,
        embed_size=EMBED_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)

    # Load model
    model_dir = os.path.join(Path_save, "{}/Original/18/read_{}/M_{}".format(mode, index[0], str(Ranges[0])))
    model_filename = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'fxy' in f][-1]
    model_path = os.path.join(model_dir, model_filename)
    model.load_state_dict(torch.load(model_path, map_location=device))

    stega_path = os.path.join(Path_save, "{}/Original/18/read_{}/M_{}/stego_sequences".format(mode, index[0], Ranges[0]))
    sequences_file = os.path.join(stega_path, 'adg_sequences.txt')
    with open(sequences_file, 'r', encoding='utf8') as f:
        stego_sequences = [line.strip() for line in f.readlines()]

    model.eval()
    extracted_bits_list = []
    for seq_idx, stego_sequence in enumerate(stego_sequences):
        print('Extracting from sequence {}'.format(seq_idx + 1))
        extracted_bits = ''
        x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
        stego_sequence_tokens = stego_sequence.split(' ') + ['_EOS']

        for gen_token in stego_sequence_tokens:
            
            log_prob = model(x)
            prob = torch.exp(log_prob)[:, -1, :].reshape(-1)
            prob[1] = 0
            prob = prob / prob.sum()
            prob, indices = prob.sort(descending=True)
            if gen_token == '_EOS':
                break # EOS token is not in the vocabulary
            gen_token_idx = vocabulary.w2i.get(gen_token, None)
            if gen_token_idx is None:
                print(f'Token {gen_token} not found in vocabulary')
                break

            if prob[0].item() > 0.5:
                # gen_token_idx = vocabulary.w2i.get(gen_token, None)
                
                # if gen_token_idx == indices[0]:
                #     # 不嵌入比特
                #     print('gen_token_idx == {} '.format(gen_token_idx))
                #     continue
                if gen_token_idx != indices[0].item():
                    print(f'Token mismatch: expected {vocabulary.i2w[indices[0].item()]} but got {gen_token}')
                    break
            else:
                bit_tmp = 0
                bits_extracted = ''
                while prob[0] <= 0.5:
                    bit = 1
                    while (1 / 2 ** (bit + 1)) > prob[0]:
                        bit += 1
                    mean = 1 / 2 ** bit
                    prob_list = prob.tolist()
                    indices_list = indices.tolist()
                    result = []
                    for j in range(2 ** bit):
                        result.append([[], []])
                    for j in range(2 ** bit - 1):
                        result[j][0].append(prob_list[0])
                        result[j][1].append(indices_list[0])
                        del (prob_list[0])
                        del (indices_list[0])
                        while sum(result[j][0]) < mean:
                            delta = mean - sum(result[j][0])
                            idx = near(prob_list, delta)
                            if prob_list[idx] - delta < delta:
                                result[j][0].append(prob_list[idx])
                                result[j][1].append(indices_list[idx])
                                del (prob_list[idx])
                                del (indices_list[idx])
                            else:
                                break
                            if len(prob_list) == 0:
                                break
                        mean = sum(prob_list) / (2 ** bit - j - 1) if (2 ** bit - j - 1) > 0 else 0
                    result[2 ** bit - 1][0].extend(prob_list)
                    result[2 ** bit - 1][1].extend(indices_list)

                    # gen_token_idx = vocabulary.w2i.get(gen_token, None)
                    # if gen_token_idx is None:
                    #     break
                    group_found = False
                    for group_idx, group in enumerate(result):
                        if gen_token_idx in group[1]:
                            int_embed = group_idx
                            bits = int2bits(int_embed, bit)
                            bits_extracted += ''.join(str(b) for b in bits)
                            print(f"Extracted bits: {bits}, int_embed: {int_embed}, token: {gen_token}")

                            group_found = True
                            prob = torch.FloatTensor(group[0]).to(device)
                            indices = torch.LongTensor(group[1]).to(device)
                            prob = prob / prob.sum()
                            prob, _ = prob.sort(descending=True)
                            indices = indices[_]
                            # bit_tmp += bit
                            break
                    if not group_found:
                        print(f'Token {gen_token} not found in any group')
                        break

                extracted_bits += bits_extracted
            gen_token_idx = vocabulary.w2i.get(gen_token, None)
            if gen_token_idx is None:
                break
            x = torch.cat([x, torch.LongTensor([[gen_token_idx]]).to(device)], dim=1).to(device)
            if gen_token == '_EOS':
                break

        extracted_bits_list.append(extracted_bits)

    extracted_bits_file = os.path.join(stega_path, 'extracted_bits.txt')
    with open(extracted_bits_file, 'w', encoding='utf8') as f:
        for bits in extracted_bits_list:
            f.write(bits + '\n')
    print('Extracted bits saved to {}'.format(extracted_bits_file))
