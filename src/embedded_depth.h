#pragma once
#include <Python.h>
#include <sensor_msgs/Image.h>
#include <ros/ros.h>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstring>

class EmbeddedDepth {
    struct Item {size_t index; sensor_msgs::ImageConstPtr image; long long queued;};
    std::mutex mutex_;std::condition_variable cv_;std::deque<Item> pending_;
    bool stop_=false,ready_=false,failed_=false;size_t received_=0,peak_=0;int interval_;
    std::thread thread_;std::function<void(sensor_msgs::ImageConstPtr)> deliver_;
    std::ofstream frames_,inputs_;std::string output_;
    static long long now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    void work(std::string weights,std::string mask){
      try {
        static wchar_t home[]=ODGS_PYTHON_HOME;Py_SetPythonHome(home);
        Py_Initialize();
        PyObject* module=PyImport_ImportModule("worker");
        if(!module)throw std::runtime_error("Embedded DAP import failed");
        PyObject* init=PyObject_GetAttrString(module,"initialize");
        PyObject* initialized=PyObject_CallFunction(init,"ss",weights.c_str(),mask.c_str());Py_DECREF(init);
        if(!initialized)throw std::runtime_error("Embedded DAP initialization failed");Py_DECREF(initialized);
        PyObject* infer=PyObject_GetAttrString(module,"predict");
        {std::lock_guard<std::mutex> lock(mutex_);ready_=true;}
        std::ofstream(output_+"/ready.json")<<"{\"monotonic_ns\":"<<now()<<"}";
        cv_.notify_all();
        while(true){
          Item item;
          {std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return stop_||!pending_.empty();});if(pending_.empty())break;item=pending_.front();pending_.pop_front();}
          const auto& m=*item.image;const auto start=now();
          if(m.encoding!="bgr8")throw std::runtime_error("Embedded DAP expects bgr8");
          PyObject* memory=PyMemoryView_FromMemory(reinterpret_cast<char*>(const_cast<unsigned char*>(m.data.data())),m.data.size(),PyBUF_READ);
          PyObject* result=PyObject_CallFunction(infer,"Oiii",memory,m.height,m.width,m.step);Py_DECREF(memory);
          if(!result)throw std::runtime_error("Embedded DAP prediction failed");
          if(!PyBytes_Check(result)||PyBytes_Size(result)!=static_cast<Py_ssize_t>(m.height*m.width*sizeof(float)))throw std::runtime_error("Embedded depth shape mismatch");
          auto depth=boost::make_shared<sensor_msgs::Image>();depth->header=m.header;depth->height=m.height;depth->width=m.width;depth->step=m.width*sizeof(float);depth->encoding="32FC1";
          const auto* bytes=PyBytes_AsString(result);depth->data.assign(bytes,bytes+PyBytes_Size(result));Py_DECREF(result);
          deliver_(depth);const auto end=now();
          {std::lock_guard<std::mutex> lock(mutex_);frames_<<item.index<<','<<m.header.stamp.toNSec()<<','<<item.queued<<','<<start<<','<<end<<','<<pending_.size()<<std::endl;}
        }
        Py_DECREF(infer);Py_DECREF(module);
        PyEval_SaveThread();
      }catch(const std::exception& e){
        if(PyErr_Occurred())PyErr_Print();ROS_ERROR_STREAM(e.what());
        std::ofstream(output_+"/failure.json")<<"{\"error\":\"Embedded DAP failed; see mapping.log\"}";
        {std::lock_guard<std::mutex> lock(mutex_);failed_=true;}cv_.notify_all();ros::shutdown();
      }
    }
public:
    EmbeddedDepth(const std::string& output,const std::string& weights,const std::string& mask,std::function<void(sensor_msgs::ImageConstPtr)> deliver,int interval):interval_(interval),deliver_(deliver),output_(output){
      std::filesystem::create_directories(output);frames_.open(output+"/frames.csv");frames_<<"index,stamp_ns,queued_ns,start_ns,end_ns,pending\n";
      inputs_.open(output+"/inputs.csv");inputs_<<"index,stamp_ns,received_ns\n";
      thread_=std::thread(&EmbeddedDepth::work,this,weights,mask);
      std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return ready_||failed_;});
      if(failed_){lock.unlock();thread_.join();throw std::runtime_error("Embedded depth startup failed");}
    }
    void push(const sensor_msgs::ImageConstPtr& image){
      std::lock_guard<std::mutex> lock(mutex_);const auto i=received_++;
      inputs_<<i<<','<<image->header.stamp.toNSec()<<','<<now()<<std::endl;
      if((i+1)%interval_==0){pending_.push_back({i,image,now()});peak_=std::max(peak_,pending_.size());cv_.notify_one();}
    }
    ~EmbeddedDepth(){
      {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}cv_.notify_one();thread_.join();
      std::ofstream(output_+"/queue.json")<<"{\"received\":"<<received_<<",\"peak_pending\":"<<peak_<<",\"pending\":"<<pending_.size()<<"}";
    }
};
