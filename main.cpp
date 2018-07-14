#include <iostream>

#include <caffe/net.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <iomanip>

void init_images_labels(std::vector<float> &images, std::vector<float> &labels, const std::string &images_path,
                        const std::string &labels_path);


int main()
{
    using namespace std;
    //the resource paths
    string folder = "../resources/";
    string network_path = folder + "network.prototxt";
    string images_path = folder + "train-images-idx3-ubyte";
    string labels_path = folder + "train-labels-idx1-ubyte";

    //the actual net
    caffe::Net<float> net(network_path, caffe::Phase::TRAIN);

    //initialise the input layer
    caffe::Layer<float>* im = net.layer_by_name("images").get();

    auto * dataLay = dynamic_cast<caffe::MemoryDataLayer<float>*>(im);
    vector<float> images;
    vector<float> labels;
    init_images_labels(images, labels, images_path, labels_path);

    dataLay->Reset(images.data(), labels.data(), (int)images.size());

    net.Forward();
    for(const auto &b: net.blobs())
    {
        int size = 1;
        for(auto s: b->shape())
        {
            size *= s;
        }/*
        for(int i = 0; i<size; i++)
        {
            cout << setw(3)<< b->cpu_data()[i] << " ";
            if((i+1)%28 == 0)
                cout << endl;
        }*/
        cout << endl << size << endl;
    }


    return 0;
}
void init_images_labels(std::vector<float> &images, std::vector<float> &labels, const std::string &images_path,
                        const std::string& labels_path)
{
    auto image_file = std::ifstream(images_path, std::ios::binary);
    auto label_file = std::ifstream(labels_path, std::ios::binary);
    const int num = 60000;
    unsigned quant = 28*28;
    unsigned char bytes_buffer[quant];
    unsigned char byte_buffer;
    images.clear();
    labels.clear();
    images.reserve(num);
    labels.reserve(num*quant);

    label_file.seekg(8);
    const float max_byte = 256;
    for(int i = 0; i<num; i++)
    {
        label_file.read(reinterpret_cast<char *>(&byte_buffer), sizeof(char));
        labels.push_back((float)byte_buffer);
    }
    image_file.seekg(16);
    for(int i = 0; i<num; i++)
    {
        image_file.read(reinterpret_cast<char *>(bytes_buffer), sizeof(bytes_buffer));
        for(int j = 0; j<quant; j++)
        {
            images.push_back(bytes_buffer[j]/max_byte);
        }
    }

}

