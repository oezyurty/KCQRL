import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text
import json

KEYS = ["UserId", "ConceptId", "QuestionId"]

def read_data_from_csv(read_file, write_file, dq2c):
    stares = []

    df = pd.read_csv(read_file, low_memory=False)
    cs = []
    for i, row in df.iterrows():
        qid = row["QuestionId"]
        cid = dq2c[qid]
        cs.append(cid)
    df["ConceptId"] = cs

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df = df.dropna(subset=["UserId", "DateAnswered", "QuestionId", "IsCorrect"])

    df['IsCorrect'] = df['IsCorrect'].astype(int)
    df.loc[:, "DateAnswered"] = df.loc[:, "DateAnswered"].apply(lambda t: change2timestamp(t))

    # IMPORTANT, WE FILTER OUT THE RETRIED ATTEMPTS HERE
    df = df[df['AnswerType'].isin(['Checkin', 'Checkout'])]

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    ui_df = df.groupby(['UserId'], sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["DateAnswered", "AnswerId"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['ConceptId'].astype(str)
        seq_ans = tmp_inter['IsCorrect'].astype(str)
        seq_problems = tmp_inter['QuestionId'].astype(str)
        seq_start_time = tmp_inter['DateAnswered'].astype(str)
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return


def load_q2c(fname):
    dq2c = dict()

    df = pd.read_csv(fname)

    for i in range(len(df)):
        dq2c[df.QuestionId[i]] = df.ConstructId[i]

    return dq2c