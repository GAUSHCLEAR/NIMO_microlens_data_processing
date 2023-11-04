import streamlit as st
from data_processing import process_one_peak_file_list,plot_data, save_uploaded_file, plot_2_data
import os

# 如果没有tmp_file1和tmp_file2文件夹，就创建



st.title("微透镜测量比较工具")

col1, col2= st.columns(2)
file_list1=col1.file_uploader("上传第一组微透镜测量文件", type=['xlsx', 'csv'],                     accept_multiple_files=True)
file_list2=col2.file_uploader("上传第二组微透镜测量文件", type=['xlsx', 'csv'],
                            accept_multiple_files=True) 


if file_list1:
    if not os.path.exists("tmp_file1"):
        os.makedirs("tmp_file1")
    for file in file_list1:
        save_uploaded_file(file, "tmp_file1")
    data_list1=[os.path.join("tmp_file1", file.name) for file in file_list1]
    data1=process_one_peak_file_list(data_list1)
    plt1=plot_data(data1, "1")
    col1.pyplot(plt1)
    # 删除临时文件
    for file in file_list1:
        os.remove(os.path.join("tmp_file1", file.name))

if file_list2:
    if not os.path.exists("tmp_file2"):
        os.makedirs("tmp_file2")    
    for file in file_list2:
        save_uploaded_file(file, "tmp_file2")
    data_list2=[os.path.join("tmp_file2", file.name) for file in file_list2]
    data2=process_one_peak_file_list(data_list2)
    plt2=plot_data(data2, "2")
    col2.pyplot(plt2)
    # 删除临时文件
    for file in file_list2:
        os.remove(os.path.join("tmp_file2", file.name))

if file_list1 and file_list2:
    left,center,right=st.columns([4,1,4])
    compare=st.button("比较")
    if compare:
        plt_compare=plot_2_data(data1, data2)
        st.pyplot(plt_compare)

        

