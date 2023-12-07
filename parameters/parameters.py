# -*- coding: UTF-8 -*-

## 接受可变长参数
def avg(first,*rest):
    return first+sum(rest)

## 接受字典参数
def display_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

## 关键字参数
def greet(name, message="Hello"):
    print(f"{message}, {name}!")

if __name__ == '__main__':
    print(avg(1,2,3))
    display_info(name='1',age=1)
    greet("cr7",message="你好")
