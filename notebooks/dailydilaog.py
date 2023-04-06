import pandas as pd
import tools
with open("/workspaces/codespaces-jupyter/data/ijcnlp_dailydialog/dialogues_text.txt", "r") as f:
    raw_data = f.read()
s = [x.split("__eou__") for x in raw_data.split("\n")]
q = []
for i in s:
    q.extend(i)
dataset = [i for i in q if i != '']
data = []
for chank_num in range(0, len(dataset)//1000):
    try:
        date_sentces = tools.extract_date_roots(dataset[chank_num*1000:(chank_num+1)*1000])
        for i in date_sentces:
            if (i[1] == None):
                data.append([i[0], i[1], i[2], 0])
            else:
                data.append([i[0], i[1], i[2], 1])
        print(f"chunk {chank_num} completed")
    except Exception as e:
        print(f"chunk failed {chank_num}. Exception: {e}")
data_df = pd.DataFrame(data, columns=["text", "root", "event", "label"])
data_df.to_csv("../processed_data/dailydialog_datesentences.csv")
