import os
from logging import basicConfig, getLogger

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = getLogger(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

mapping = {
    0: "含意",
    1: "中立",
    2: "矛盾",
}


def load_data(data_path):
    # pandasでcsvファイルを読み込む
    df = pd.read_csv(data_path)
    # データのなかの60行から120行を取得
    trunc_df = df[60:120]
    # idnexを振り直す
    trunc_df = trunc_df.reset_index(drop=True)
    return trunc_df


def main():
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("model loading...")
    model = genai.GenerativeModel("gemini-pro")
    df = load_data("data/data.csv")

    for i in range(len(df)):
        query = f"#参考# 以下の例を参考に判定しなさい\n #前提文例#\n 上手な人が後輪走行で自転車に乗っている\n #仮定文例# \nある人が後輪走行でその自転車に乗っている \n#回答例# 判定:0.95 \n#前提文例#\n 庭にいる男の子たちのグループが遊んでいて、男性が後ろの方に立っている\n #仮定文例#\n 幼い男の子たちが戸外で遊んでいて、その男性が近くで微笑んでいる\n #回答例# 判定:0.5\n #前提文例#\n 茶色の犬が、ズボンをはいた男性の前にいる別の動物に襲いかかっている\n #仮定文例#\n 茶色の犬が、ズボンをはいた男性の前にいる別の動物を助けている\n #回答例# 判定:0.05\n#指示#\n 前提文と仮定文の関係がどの程度含意しているかを判定してもらいたい。 含意度を1.00～0.00の間で判定せよ。1.00に近いほど含意している可能性が高く、0.00に近いほど矛盾している可能性が高い。\n 次の項目のみを返答せよ。 判定:[含意度]\n #前提# {df['s1'][i]}\n #仮定# {df['s2'][i]}"

        response = model.generate_content(query)
        logger.info(f"====================================")
        logger.info(f"前提: {df['s1'][i]}")
        logger.info(f"仮定: {df['s2'][i]}")
        logger.info(f"label : {mapping[df['label'][i]]}")
        logger.info(f"======= Gemini Pro さんの判定 =======")
        logger.info(response.text)
        logger.info(f"====================================")


if __name__ == "__main__":
    main()
