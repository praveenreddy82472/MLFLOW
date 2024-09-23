import argparse

if __name__ == '__main__':
    
    args= argparse.ArgumentParser()
    
    args.add_argument('--name',"-n", type=str,default="Praveen")
    args.add_argument("--age","-a", type=float,default=23.0)
    parse_args= args.parse_args()
    
    
    print(parse_args.name,parse_args.age)
    
    