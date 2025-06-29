import cv2 as cv
from max.driver import CPU, Accelerator, Tensor, accelerator_count, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops, DeviceRef, TensorValue
from pathlib import Path
import numpy as np
from time import perf_counter

def gaussian_blur(device: Device, image: TensorValue, kernel_size: int=5, sigma: float=1., padding: bool=False) -> TensorValue:
    N = kernel_size

    # generate the kernel
    ax = np.linspace(-(N - 1) / 2., (N - 1) / 2., N)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    arr =  (kernel / np.sum(kernel))

    # reshape the expected RSCF layout
    kernel = ops.constant(arr, dtype=image.dtype, device=DeviceRef.from_device(device))
    kernel = kernel.reshape((N, N, 1, 1))
    
    pad_value = (N // 2) if padding else 0

    return ops.conv2d(
        # expects NHWC layout but we only ever have 1 input image
        image.reshape([1] + image.shape.static_dims),
        kernel,
        padding=[pad_value] * 4
    )[0].tensor

def sobel_filter(device: Device, image: TensorValue) -> TensorValue:

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32).reshape(3, 3, 1, 1)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32).reshape(3, 3, 1, 1)

    grad_x = ops.conv2d(
        # expects NHWC layout but we only ever have 1 input image
        image.reshape([1] + image.shape.static_dims),
        Kx,
    )[0].tensor

    grad_y = ops.conv2d(
        # expects NHWC layout but we only ever have 1 input image
        image.reshape([1] + image.shape.static_dims),
        Ky,
    )[0].tensor

    mag = ops.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = ops.custom(
        name="atan2",
        device=DeviceRef.from_device(device),
        values=[
            grad_x,
            grad_y
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape, device=DeviceRef.from_device(device))],
    )[0].tensor

    return mag, direction


if __name__ == "__main__":
    img_orig = cv.imread("bucky_birthday.jpeg")
    
    # img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    device: Device
    try:
        device = Accelerator()
    except:
        device = CPU()

    img = Tensor.from_numpy(img_orig).to(device)

    input_types = [TensorType(dtype=img.dtype, shape=img.shape, device=DeviceRef.from_device(device))]
    output_types = input_types 

    with Graph(
        "canny",
        input_types = input_types,
        output_types = output_types,
        custom_extensions = [Path(__file__).parent / "operations"]
        ) as graph:
            im = graph.inputs[0]

            # normalize the image
            im = im.cast(DType.float32)
            im /= 255.0

            im = ops.custom(
                name='grayscale',
                device=DeviceRef.from_device(device),
                values=[
                    im
                ],
                out_types=[TensorType(dtype=im.dtype, shape=(im.shape[0], im.shape[1]), device=DeviceRef.from_device(device))],
            )[0].tensor

            im = im.reshape(im.shape.static_dims + [1])

            
            im = gaussian_blur(device, im)

            # apply sobel operation
            mag, direction = sobel_filter(device, im)

            # non max supression
            im = ops.custom(
                name="non_max_supression",
                device=DeviceRef.from_device(device),
                values=[
                    mag,
                    direction
                ],
                out_types=[TensorType(dtype=im.dtype, shape=im.shape, device=DeviceRef.from_device(device))],
            )[0].tensor

            # restore image as the next two steps don't involve any fancy arithmetic
            im = ops.max(ops.min(im, 1.0), 0.0)
            im = ops.cast(im * 255.0, dtype=DType.uint8)

            # apply threshold values
            im = ops.custom(
                name="init_thresholding",
                device=DeviceRef.from_device(device),
                values=[
                    im,
                    ops.constant(100, dtype=im.dtype, device=DeviceRef.from_device(CPU())),
                    ops.constant(200, dtype=im.dtype, device=DeviceRef.from_device(CPU()))
                ],
                out_types=[TensorType(dtype=im.dtype, shape=im.shape, device=DeviceRef.from_device(device))],
            )[0].tensor

            # This is a dependent operation so repeat it until it converges
            # TODO: fold this logic into the kernel
            while True:
                # drop any weak edges without strong connections
                t = ops.custom(
                    name="thresholding",
                    device=DeviceRef.from_device(device),
                    values=[
                        im
                    ],
                    out_types=[TensorType(dtype=im.dtype, shape=im.shape, device=DeviceRef.from_device(device))],
                )[0].tensor

                if t == im:
                    break
                im = t

            graph.output(im)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    
    start = perf_counter()
    res = model.execute(img)[0].to(CPU()).to_numpy().copy()
    end = perf_counter()

    print('MAX graph took:', end-start)

    
    start = perf_counter()
    c = cv.Canny(img_orig, 100, 200)
    end = perf_counter()
    print('OpenCV took:', end-start)
    cv.imshow('res',res)
    cv.imshow('ocv', c)
    cv.waitKey(0)
    cv.destroyAllWindows()
