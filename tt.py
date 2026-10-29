import kagglehub

# Download latest version
path = kagglehub.competition_download('m5-forecasting-accuracy')

print("Path to competition files:", path)
