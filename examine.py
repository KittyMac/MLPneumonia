from __future__ import division

# Note: we're trying out PlaidML
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# Note: we're using code in both
import sys
sys.path.insert(0, './pneumonia_classification')
sys.path.insert(0, './pneumonia_localization')

from data import kPatientID
from data import kBoundsX
from data import kBoundsY
from data import kBoundsWidth
from data import kBoundsHeight
from data import kTarget


def GetAllPatientInfo():
	patientInfo = []
	with open("pneumonia_classification/data/stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	return patientInfo

def Examine(dcmFilePath):
	# Examine the patient's xray from the supplied path, returning the bounding boxes of any cases of penumonia discovered
	print("Examine()", dcmFilePath)
	
	
def TestPatient(patientId):
	# load the specified patient from the training data, examine their DCM, and compare the results to see how we did
	print("Test()", patientId)
	allPatients = GetAllPatientInfo()
	

def TestRandomPatients(num):
	# load random patients from the training data and test them
	print("TestRandomPatients()", num)

def GenerateSubmission():
	# load all patients from the submission testing data, examine them, and report the results in a CSV files suitable for submission to Kaggle
	print("GenerateSubmission()")

if __name__ == '__main__':
	if len(sys.argv) >= 2:
		if sys.argv[1] == "test":
			if len(sys.argv) >= 3:
				try: 
					num = int(sys.argv[2])
					TestRandomPatients(num)
					exit(0)
				except ValueError:
					TestPatient(sys.argv[2])
					exit(0)
		elif sys.argv[1] == "submit":
			GenerateSubmission()
			exit(0)

	TestRandomPatients(1)

	