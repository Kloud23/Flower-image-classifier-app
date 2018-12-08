
import sys
import json
import ImageClassifierFunctions as icfs
from torchvision import models


def predict(path_to_image="./test/10/image_07090.jpg", gpu= "False", path_to_checkpoint="./checkpoint.pth", top_number_of_prob=5, path_to_classes_json ="./cat_to_name.json"):

	print(40*"=")
	print("Prediction for your image is on the way!")
	print(40*"=")
	with open(path_to_classes_json, 'r') as f:
	    cat_to_name = json.load(f)
	print("Categories of classes loaded!")

	loaded_model= icfs.loads_checkpoint(path_to_checkpoint)
	loaded_model.idx_to_class=dict([[value,key]for key,value in icfs.image_datasets['train'].class_to_idx.items()])
	print(40*"=")

	print("model loaded successfully")
	if gpu=="True":
		gpu = "Yes"
	elif gpu == "False":
		gpu="No"
	cla, prob = icfs.predict(image_path = path_to_image, model=loaded_model, topk=top_number_of_prob, gpu=gpu)

	print("Classes: ")
	print(cla)
	print(40*"=")
	print(40*"=")
	print("Probabilities:")
	
	print(prob)
	def get_names_from_classes(classes):
	    class_names = []
	    for c in classes:
	        class_names.append(cat_to_name[c])
	    return(class_names)
	print(40*"=")
	print(40*"=")
	print("Categories:")
	print(get_names_from_classes(cla))
	print(40*"=")

if __name__ == "__main__":
	try:
		path_to_checkpoint = (sys.argv[1])
		if path_to_checkpoint =="./":
			path_to_checkpoint = "./checkpoint.pth"

		path_to_image = (sys.argv[2])
		path_to_classes_json = (sys.argv[3])
		if path_to_classes_json == "./":
			path_to_classes_json == "./cat_to_name.json"
		
		top_number_of_prob = int(sys.argv[4])

		gpu = (sys.argv[5])
		predict(path_to_image=path_to_image, path_to_checkpoint=path_to_checkpoint, top_number_of_prob=top_number_of_prob, gpu=gpu)

	except IndexError:
		print("Rerun the command giving the parameters in the following fashion:\n")
		print("python3 predict.py path_to_checkpoint path_to_image path_to_classes_json top_number_of_prob gpu")
		print("Command line arguments require:\n path_to_checkpoint : path to checkpoint\n path_to_image: path to image \n path_to_classes_json: path to classes json('./cat_to_name.json')\n top_number_of_prob: number of top probabilities to return \n gpu : True or False\n")
