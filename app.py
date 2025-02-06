import streamlit as st # type: ignore
import warnings
# warnings.filterwarnings("ignore")
from ClassifyGan import *
from torch.autograd import Variable # type: ignore
from dataset import *
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
#Load the saved model discriminator
pthFile = r'../TrainedModels/DorC.pth'
D = torch.load(pthFile)
#Load the data set for the test
def data():
    testdataset = Testdataset()
    return testdataset
#Test the classification accuracy of the discriminator
def testClassify(dataloader):
    for i, batch_data in enumerate(dataloader):
        img_path=batch_data['img_path']#Obtain image path
        real_imgs = Variable(batch_data['image'].type(FloatTensor))  # real image
        # print(real_imgs)
        labels = Variable(batch_data['label'].type(LongTensor))  # Actual label
        real_pred, real_aux = D(real_imgs)
        #print('real_aux:',[real_aux.data.cpu().numpy()])
        gt = np.concatenate([labels.data.cpu().numpy()], axis=0)  # Real labels 0 and 1 0: indicates suspected oil spill 1: indicates oil spill image False label 2
        # print('Actual label',gt)
        pred = np.concatenate([real_aux.data.cpu().numpy()], axis=0)  # Is the judgment result of the real image after the discriminator, 0: represents the suspected oil spill 1: represents the oil spill image false label is 2
        # print('result',np.argmax(pred, axis=1))
        predlabels=np.argmax(pred, axis=1)
        acc = np.mean(np.argmax(pred, axis=1) == gt)  # Calculate the classification accuracy
        # print('accuracy:',acc)
    # The output discriminates as the path of oil spill image
    return predlabels

# Streamlit app
st.title("Oil Spill Detection")

saveDir = "../Input/newTest/Oil"

if not os.path.exists(saveDir):
    st.write("No such Directory")
# File uploader for a single image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image
    img = uploaded_file.read()

    # Create a path to store the image
    uploaded_file.name = "01.jpg"
    img_path = os.path.join(saveDir, uploaded_file.name)

    # Save the image to the system
    with open(img_path, "wb") as f:
        f.write(img)

    # Display a success message and show the image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    # Calling the Function
    dataloader = data()

    predictions = testClassify(dataloader)

    for i in range(predictions.size):
        if predictions[i] == 1:
            st.success("Oil Spill :)")
        else:
            st.warning("No Oil Spill :(")

