import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt


df = pd.read_csv("flickr_data.csv", escapechar="\\",skipinitialspace= True)

for t in {"taken_", "upload_"}:
  time = {key: df["date_" + t + key] for key in {"year", "month", "day", "hour", "minute"}}
  df["date_"+t[0:-1]] = pd.to_datetime(time).astype("category")
  df["date_"+t[0:-1]] = df["date_"+t[0:-1]].cat.codes
  df = df.drop(columns = ["date_" + t + key for key in time.keys()])

df = df.drop(columns=["user", "tags", "title", "date_taken"])
df = df.head(100)



kmeans = MiniBatchKMeans(batch_size=3, max_iter=10).fit(df)

df["category"] = kmeans.predict(df.head(100))

group = df.groupby(["category"])







# for index, row in df.iterrows() :
#   plt.scatter(row["lat"], row["long"], c=row["category"])

plt.show()




