import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


############ Load the face images 
def loadFaces(facePath):
    images = []
    imagePath = os.listdir(facePath)

    for path in imagePath:
        image = cv.imread(facePath + "/"  + path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = image.flatten()
        images.append(image)

    images = np.asarray(images)

    return images

############ Split into training and testing data
def trainTestSplit(images):
    trainingImages, testingImages = train_test_split(images, test_size=0.11, random_state=0)
    return trainingImages, testingImages

############ Normalize training set
def normTrainingSet(trainingImages, i=0):
    trainingImagesMean = np.average(trainingImages, axis=0)
    normTrainingImages = trainingImages - trainingImagesMean

    return normTrainingImages, trainingImagesMean

############ Covariance Matrix
def covMatrix(trainingImages, normTrainingImages):
    sizeofTrainData = trainingImages.shape[0]
    covarianceMatrix = np.dot(normTrainingImages, normTrainingImages.T)/sizeofTrainData

    return covarianceMatrix

############ Eigen Decomposition and getting the best Eigen Vectors based on K Eigen Values
def topEigenVectors(covarianceMatrix, K):
    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
    sortedEigenValues = eigenValues.argsort()[:K:-1]
    eigenVectors = eigenVectors[:,sortedEigenValues]
    eigenValues = eigenValues[sortedEigenValues]
    KEigenVectors = eigenVectors[0:, :K]

    return eigenValues, KEigenVectors

############ Eigen Faces
def getEigenFaces(normTrainingImages, eigenVectors):
    eigenFaces = np.matmul(normTrainingImages.T, eigenVectors)
    return eigenFaces

############ Normalized Eigen Faces
def normEigenFaces(eigenFaces):
    normalizedEigenFaces = np.zeros(eigenFaces.shape)
    for i in range(eigenFaces.shape[-1]):
        normalizedEigenFaces[:,i] = eigenFaces[:,i]/np.linalg.norm(eigenFaces, axis = 0)[i]
    return normalizedEigenFaces

############ Dispalay Eigen Faces
def display(eigenFacesNorm):
    eigenFacesReshaped = eigenFacesNorm.reshape((256, 256, eigenFacesNorm.shape[-1]))

    fig, ax = plt.subplots(2, 5)
    for i in range(5):
        for j in range(2):
            ax[j, i].imshow(eigenFacesReshaped[:, :, i+j], cmap = "gray")
    plt.title("Eigen Faces")
    plt.show()

############ Reconstruction
def imageReconstruction(testingImages, trainingImagesMean, eigenFacesNorm):
    normTestingImage = testingImages - trainingImagesMean
    testImageEigen = np.dot(normTestingImage, eigenFacesNorm)
    testImageEigenTranspose = np.dot(testImageEigen, eigenFacesNorm.T)
    reconstructedFace = testImageEigenTranspose + trainingImagesMean
    reshapedReconstructedFace = reconstructedFace.reshape((reconstructedFace.shape[0], 256, 256))

    for i in range(len(reconstructedFace)):
        temp = reshapedReconstructedFace[i] + abs(reshapedReconstructedFace[i].min()) 
        reshapedReconstructedFace[i] = (temp/temp.max())

    return reshapedReconstructedFace, reconstructedFace


############ Main Function
K = 30

images = loadFaces("./faces")

trainingImages, testingImages = trainTestSplit(images)

normTrainingImages, trainingImagesMean = normTrainingSet(trainingImages)

covarianceMatrix = covMatrix(trainingImages, normTrainingImages)

eigenValues, KEigenVectors = topEigenVectors(covarianceMatrix, K)

eigenFaces = getEigenFaces(normTrainingImages, KEigenVectors)

eigenFacesNorm = normEigenFaces(eigenFaces)

display(eigenFacesNorm)

reshapedReconstructedFace, reconstructedFace = imageReconstruction(testingImages, trainingImagesMean, eigenFacesNorm)

fig, ax = plt.subplots(1, 5)
for i in range(5):
    ax[i].imshow(reshapedReconstructedFace[i], cmap = "gray")
plt.title("Reconstructed Faces")
plt.show()

errors = {}

K = [10, 30, 50, 90, 100]
for k in K:
    eigenValues, KEigenVectors = topEigenVectors(covarianceMatrix, k)
    eigenFaces = getEigenFaces(normTrainingImages, KEigenVectors)
    eigenFacesNorm = normEigenFaces(eigenFaces)
    reshapedReconstructedFace, reconstructedFace = imageReconstruction(testingImages, trainingImagesMean, eigenFacesNorm)

    error = np.sum(np.square(reconstructedFace - testingImages/255))/testingImages.shape[0]
    # print(error)
    errors[k] = error

# print(errors)
y_axis = np.array(list(errors.values())).astype(np.uint)
plt.title("Error curve")
plt.plot(K, y_axis)
plt.show()
