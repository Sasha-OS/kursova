from PIL import Image
from itertools import groupby
import numpy as np
import keras
import os

if(os.path.exists("model.h5")):
    model = keras.models.load_model("model.h5")
elements_pred = []
elements_array = []
answer = ''
def analysPhoto(photo):
    # -------------------------------

    # ~~~3. Prediction~~~

    'loading image'
    image = Image.open(photo).convert("L")

    'resizing to 28 height pixels'
    w = image.size[0]
    h = image.size[1]
    r = w / h  # aspect ratio
    new_w = int(r * 28)
    new_h = 28
    new_image = image.resize((new_w, new_h))

    'converting to a numpy array'
    new_image_arr = np.array(new_image)

    'inverting the image to make background = 0'
    new_inv_image_arr = 255 - new_image_arr

    'rescaling the image'
    final_image_arr = new_inv_image_arr / 255.0

    'splitting image array into individual digit arrays using non zero columns'
    m = final_image_arr.any(0)
    out = [final_image_arr[:, [*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]

    '''
    iterating through the digit arrays to resize them to match input 
    criteria of the model = [mini_batch_size, height, width, channels]
    '''
    num_of_elements = len(out)
    elements_list = []

    for x in range(0, num_of_elements):

        img = out[x]

        # adding 0 value columns as fillers
        width = img.shape[1]
        filler = (final_image_arr.shape[0] - width) / 2

        if filler.is_integer() == False:  # odd number of filler columns
            filler_l = int(filler)
            filler_r = int(filler) + 1
        else:  # even number of filler columns
            filler_l = int(filler)
            filler_r = int(filler)

        arr_l = np.zeros((final_image_arr.shape[0], filler_l))  # left fillers
        arr_r = np.zeros((final_image_arr.shape[0], filler_r))  # right fillers

        # concatinating the left and right fillers
        help_ = np.concatenate((arr_l, img), axis=1)
        element_arr = np.concatenate((help_, arr_r), axis=1)

        element_arr.resize(28, 28, 1)  # resize array 2d to 3d

        # storing all elements in a list
        elements_list.append(element_arr)
    global elements_pred, elements_array, answer

    elements_array = np.array(elements_list)

    'reshaping to fit model input criteria'
    elements_array = elements_array.reshape(-1, 28, 28, 1)

    'predicting using the model'

    elements_pred = model.predict(elements_array)
    elements_pred = np.argmax(elements_pred, axis=1)

    # -------------------------------

    # ~~~4. Mathematical Operation~~~

    'creating the mathematical expression'
    m_exp_str = math_expression_generator(elements_pred)

    'calculating the mathematical expression using eval()'
    while True:
        try:
            answer = eval(m_exp_str)  # evaluating the answer
            answer = round(answer, 2)
            equation = m_exp_str + " = " + str(answer)
            answer = equation
            print(equation)  # printing the equation
            break

        except SyntaxError:
            print("Invalid predicted expression!!")
            print("Following is the predicted expression:")
            print(m_exp_str)
            break

    # -------------------------------

    # ~~~5. Model Update~~~

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++


def math_expression_generator(arr):
    op = {
        10,  # = "/"
        11,  # = "+"
        12,  # = "-"
        13  # = "*"
    }

    m_exp = []
    temp = []

    'creating a list separating all elements'
    for item in arr:
        if item not in op:
            temp.append(item)
        else:
            m_exp.append(temp)
            m_exp.append(item)
            temp = []
    if temp:
        m_exp.append(temp)

    'converting the elements to numbers and operators'
    i = 0
    num = 0
    for item in m_exp:
        if type(item) == list:
            if not item:
                m_exp[i] = ""
                i = i + 1
            else:
                num_len = len(item)
                for digit in item:
                    num_len = num_len - 1
                    num = num + ((10 ** num_len) * digit)
                m_exp[i] = str(num)
                num = 0
                i = i + 1
        else:
            m_exp[i] = str(item)
            m_exp[i] = m_exp[i].replace("10", "/")
            m_exp[i] = m_exp[i].replace("11", "+")
            m_exp[i] = m_exp[i].replace("12", "-")
            m_exp[i] = m_exp[i].replace("13", "*")

            i = i + 1

    'joining the list of strings to create the mathematical expression'
    separator = ' '
    m_exp_str = separator.join(m_exp)

    return m_exp_str


def model_update(X, y, model):
    from tensorflow.keras.optimizers import RMSprop
    from keras.utils.np_utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator

    y = to_categorical(y, num_classes=14)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X)

    # freezing layers 0 to 4
    for l in range(0, 5):
        model.layers[l].trainable = False

    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        datagen.flow(X, y, batch_size=1),
        epochs=50,
        verbose=1
    )

    'saving the model'
    model.save("updated_model.h5")

    print("Model has been updated!!")


def updateModel(corr_ans_str):
    global elements_pred, elements_array
    corr_ans_str = corr_ans_str.replace(" ", "")

    def feedback_conversion(correct_ans_str):
        return [char for char in correct_ans_str]

    corr_ans_list = feedback_conversion(corr_ans_str)
    dic = {"/": "10", "+": "11", "-": "12", "*": "13"}
    corr_ans_list = [dic.get(n, n) for n in corr_ans_list]
    corr_ans_arr = np.array(list(map(int, corr_ans_list)))
    print(corr_ans_arr.shape)

    'comparing the expressions and getting the indexes of the wrong predictioned elements'
    wrong_pred_indices = []

    for i in range(len(corr_ans_arr)):
        if corr_ans_arr[i] == elements_pred[i]:
            pass
        else:
            wrong_pred_indices.append(i)

    'picking up the wrongly predicted elements'
    X = elements_array[[wrong_pred_indices]]

    'reshaping to fit model input standards'
    if len(X.shape) == 3:
        X = X.reshape(-1, 28, 28, 1)
    else:
        pass

    'the correct answers as labels'
    y = corr_ans_arr[[wrong_pred_indices]]

    'updating the model'
    model_update(X, y, model)
