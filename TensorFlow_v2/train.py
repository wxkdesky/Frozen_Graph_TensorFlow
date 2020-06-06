import tensorflow as tf
from tensorflow import keras
import os,sys,numpy as np
from tensorflow.keras.layers import Input,Dense,Concatenate
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def main():

    tf.random.set_seed(seed=0)

    a=Input(shape=(5,),name='a')
    b=Input(shape=(10,),name='b')
    a1=Dense(10)(a)
    b1=Dense(10)(b)
    c=Concatenate()([a1,b1])
    d=Dense(2)(c)
    model=Model(inputs=[a,b],outputs=[d])
    model.compile(loss='mae',optimizer='adam')
    model.summary()
    input1=np.random.uniform(size=(100,5))
    input2=np.random.uniform(size=(100,10))
    label=np.random.uniform(size=(100,2))

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    model.compile(optimizer="adam",
                  loss="mae",
                  metrics=["accuracy"])

    # Train model
    model.fit([input1,input2],label, epochs=1)

    # Save model to h5 format
    model.save("model.hdf5")
    # Save model to SavedModel format
    model.save("model.tf")

    # Convert Keras model to ConcreteFunction
    @tf.function
    def nn(*args):
        return model(args)
    full_model = nn.get_concrete_function(
        [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype,name='a'),
         tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype,name='b')])

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./",
                      name="frozen_graph.pb",
                      as_text=False)


if __name__ == "__main__":

    main()
