import pandas as pd
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from matplotlib import pyplot as plt


df = pd.read_csv("flickr_data.csv", escapechar="\\",skipinitialspace= True)

for t in {"taken_", "upload_"}:
  time = {key: df["date_" + t + key] for key in {"year", "month", "day", "hour", "minute"}}
  df["date_"+t[0:-1]] = pd.to_datetime(time).astype("category")
  df["date_"+t[0:-1]] = df["date_"+t[0:-1]].cat.codes
  df = df.drop(columns = ["date_" + t + key for key in time.keys()])

df = df.drop(columns=["id", "user", "tags", "title", "date_taken", "date_upload"])
df  = df.drop_duplicates(subset=None, keep='first', inplace=False)

#normalize the data, for time 
#remove doubles

kmeans = MiniBatchKMeans(batch_size=100000, max_iter=1000000, n_clusters=12).fit(df)
# kmeans = DBSCAN(eps=3,min_samples=2).fit(df);
BBox = ((df["lat"].min(),   df["lat"].max(), df["long"].min(), df["long"].max()))

df["category"] = kmeans.predict(df)

groups = df.groupby(["category"])

image = plt.imread('map_lyon.png')

for name, group in groups:
  plt.scatter(group["lat"],group["long"], s=1)

# for index, row in df.iterrows() :
#   plt.scatter(row["lat"], row["long"], c=row["category"])

plt.imshow(image, zorder=0, extent = BBox, aspect= 'equal')

plt.show()




