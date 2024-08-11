import gzip
import json
import copy

path = "data/fbank/gigaspeech_cuts_M-sp1_1_fbank.jsonl.gz"
disc_path = "data/fbank/gigaspeech_cuts_M-sp1_1.jsonl.gz"
transform_path = "data/fbank/gigaspeech_cuts_M-sp1_1_fbank_disc_p.jsonl.gz"

# path = "data/fbank/gigaspeech_cuts_DEV_fbank.jsonl.gz"
# disc_path = "data/fbank/gigaspeech_cuts_DEV.jsonl.gz"
# transform_path = "data/fbank/gigaspeech_cuts_DEV_fbank_disc_test.jsonl.gz"


def process_files(input_path, disc_path, output_path):
    with gzip.open(input_path, 'rt', encoding='utf-8') as data_file, \
         gzip.open(disc_path, 'rt', encoding='utf-8') as disc_file, \
         gzip.open(output_path, 'wt', encoding='utf-8') as out_file:
        
        data_iter = iter(data_file)
        previous_data = None
        for disc_line in disc_file:
            current_data = json.loads(disc_line)
            try:
                if previous_data is None:
                    previous_data = json.loads(next(data_iter))
                    tmp = copy.deepcopy(previous_data)
                    tmp['id'] = tmp['id'] + '_' + current_data['id']
                    tmp['supervisions'][0]['id'] = tmp['id']
                    current_data['custom']['previous_data'] = tmp
                else:
                    previous_id = previous_data.get('id', '')
                    current_id = current_data.get('id', '')
                    previous_dialog = previous_id.strip().split('_')[0]
                    current_dialog = current_id.strip().split('_')[0]
                    
                    if previous_dialog == current_dialog:
                        # 不需要深拷贝，直接赋值即可，因为previous_data之后不会再用
                        previous_data['id'] = previous_data['id'] + '_' + current_data['id']
                        previous_data['supervisions'][0]['id'] = previous_data['id']
                        current_data['custom']['previous_data'] = previous_data
                        previous_data = json.loads(next(data_iter))
                    else:
                        previous_data['id'] = previous_data['id'] + '_' + current_data['id']
                        previous_data['supervisions'][0]['id'] = previous_data['id']
                        current_data['custom']['previous_data'] = previous_data
                        previous_data = json.loads(next(data_iter))
                    
            except StopIteration:
                # data_iter耗尽，处理剩余的disc_file中的行
                previous_data['id'] = previous_data['id'] + '_' + current_data['id']
                previous_data['supervisions'][0]['id'] = previous_data['id']
                current_data['custom']['previous_data'] = previous_data
            
            out_file.write(json.dumps(current_data) + '\n')

# 调用函数处理文件
process_files(path, disc_path, transform_path)
