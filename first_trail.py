import mlflow

def sum(a,b):
    return a * b



if  __name__ == '__main__':
    with mlflow.start_run():
        a,b = 10,20
        z = sum(a,b)
        mlflow.log_param('a',a)
        mlflow.log_param('b',b)
        mlflow.log_param('z',z)
    print(f'Sum of a and b is {z}')