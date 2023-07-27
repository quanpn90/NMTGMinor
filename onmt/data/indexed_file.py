
import io
import torch

def add_data(file_data, file_index, index, data):
    pos = file_data.tell()

    data = data.to(torch.float16).cpu()
    buffer = io.BytesIO()
    torch.save(data, buffer)
    bytes = buffer.getbuffer().tobytes()

    file_data.write(bytes)
    file_index[index] = (pos,len(bytes))

def read_data(file_data, file_index, index):
    if index not in file_index:
        #print("WARNING: Features not found")
        return None
    pos, length = file_index[index]
    file_data.seek(pos)
    bytes = file_data.read(length)
    buffer = io.BytesIO(bytes)
    data = torch.load(buffer, map_location="cpu")
    return data

if __name__ == "__main__":
    name = "test"

    file_data = open(name + ".data","wb")
    file_index = {}

    index = 42
    data = torch.randn(10,15)
    print(data)

    add_data(file_data, file_index, index, data)

    file_data.close()
    torch.save(file_index, name + ".index")

    file_data = open(name + ".data", "rb")
    file_index = torch.load(name + ".index")

    data = read_data(file_data, file_index, index)
    print(data)