import argparse
import os
import re
import regex
from tqdm import tqdm
from pathlib import Path
import jieba
from common import *
setup_stderr_logger()
from common.file_utils import FileUtils
from common.text_utils.tokenization import Tokenizer


DW_PATH = Path('/data/common/dw_parsed')
OUTPUT_PATH = '/tmp'


def initialize_lut(dump_date: str) -> dict:
    df_major = FileUtils.read_data_frame(DW_PATH.glob(f'major.{dump_date}.*.parquet*'))
    df_major = df_major['name_std'].dropna().to_dict()

    df_school = FileUtils.read_data_frame(DW_PATH.glob(f'school.{dump_date}.*.parquet*'))
    df_school = df_school['name_std'].dropna().to_dict()

    return {
        'major': df_major,
        'school': df_school
    }


def main(dictionary: str, path: str) -> NoReturn:
    # Pre-compute data lookup tables.
    dump_date = regex.search(r'(?<=^[a-z_]+\.).*?(?=\.)', os.path.basename(path)).group()
    segment_luts = initialize_lut(dump_date)

    # Initialize jieba.
    jieba.initialize(f'/data/common/jieba_{dictionary}_big_dict.txt')
    logger.info(f'Jieba FREQ: {len(jieba.dt.FREQ)}')

    tencent = FileUtils.read_obj(f'/data/common/tencent_{dictionary}_keys.pkl')
    logger.info(f'Dict size: {len(tencent)}')
    for word in tqdm(tencent):
        if len(word) > 1:
            jieba.add_word(word)

    logger.info(f'Jieba FREQ: {len(jieba.dt.FREQ)}')

    # Initialize tokenization subsystem.
    tokenizer = Tokenizer(f'/data/common/tencent_{dictionary}_keys.pkl', 256)

    # Process files.
    df = FileUtils.read_data_frame(path)

    # Expand standardized columns.
    for c, lut in segment_luts.items():
        if c not in df.columns:
            continue
        s = df[c]
        logger.debug(f'Expanding column "{s.name}" via lookup.')
        s = s.map(lambda x: lut.get(x, x), na_action='ignore')
        s = s.map(lambda x: x if len(x) > 0 else None, na_action='ignore')
        df[f'{c}_std'] = s

    for c in df.columns:
        s = df[c]
        if s.dtype != 'object' or s.name == 'industry_ids':
            continue
        logger.debug(f'Processing "{s.name}" directly.')
        lut = set(s.dropna())
        lut = {k: tokenizer.apply(jieba.cut(k)) for k in tqdm(lut)}
        s = s.map(lambda x: lut[x], na_action='ignore')
        df[c] = s

    # Write output file.
    path = os.path.basename(path)
    path = re.search(r'.*\.parquet', path).group()
    path = os.path.join(OUTPUT_PATH, path)
    FileUtils.write_data_frame(path, df)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dictionary', type=str, help='The dictionary to be used.')
    args.add_argument('files', nargs='*')
    args = args.parse_args()
    assert args.dictionary is not None
    for path in args.files:
        main(args.dictionary, path)