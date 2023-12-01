# from pymongo import MongoClient
# from PIL import Image
# import io

# # Connect to MongoDB (make sure your MongoDB server is running)
# client = MongoClient('mongodb://localhost:27017/')
# db = client['my_database']  # Choose or create a database
# collection = db['ALPR system']   # Choose or create a collection

# # Load an image from file (replace 'path/to/image.jpg' with your image path)
# image_path = 'path/to/image.jpg'
# image = Image.open(image_path)

# # Convert the image to bytes
# image_bytes = io.BytesIO()
# image.save(image_bytes, format='JPEG')
# image_bytes = image_bytes.getvalue()

# # Create a dictionary to represent the image
# image_doc = {
#     'name': 'MyImage',  # Add metadata or name for the image
#     'data': image_bytes  # Store image bytes in MongoDB
# }

# # Insert the image data into the MongoDB collection
# result = collection.insert_one(image_doc)
# print(f"Image saved to MongoDB with ID: {result.inserted_id}")q

from pymongo import MongoClient
from  datetime import datetime
from pathlib import Path

#Creating a pymongo client
# datetime_now = datetime.now() # pass this to a MongoDB doc
# print ("datetime_now:", datetime_now)

connection_string = "mongodb+srv://himanshibaghel001:9E7WxrkgMJkzu2Te@cluster0.sqjrpnc.mongodb.net/"
client = MongoClient(connection_string)
mydb = client['my_database']
mycol = mydb['my_collection']

with open("save_dir.txt", "r") as file:
        content = file.read()
        print("path: ",content)


content = Path(content)  # Example content as a Path object
txt_path = content / 'labels/realmonitor_channel_1_subtype_0'        
time_path =content / 'timestamp/realmonitor_channel_1_subtype_0'
crop_path =content / 'crops/license_plate/realmonitor_channel_1_subtype_0'
print("txt_path: ",txt_path)
print("crop_path: ",crop_path)
print("time_path: ",time_path)
crop_count = 2
while(True):
    
    try:
        with open(f'{txt_path}{crop_count}.txt', "r") as file:
            txt = file.read()
            print("txt : ",txt)
        with open(f'{time_path}{crop_count}.txt', "r") as file:
            tym = file.read()
            print("time : ",tym)
        print("count value : ",crop_count)     
        mylist = {"_id":f'license_plate_{crop_count}',"license_plate_number":txt,"datetime(Y M D H M S)":tym}
        mycol.insert_one(mylist)    
        crop_count +=1 
        # print("crop_count value : ",crop_count) 
        # print("try")
    except: 
        break
        # print("seaching ",f'{txt_path}{crop_count}.txt')
             


    # mylist = {"_id":12,"license_plate_number":content,"speed(Km/h)":60,"datetime(Y M D H M S)":datetime_now}
    # mycol.insert_one(mylist)








