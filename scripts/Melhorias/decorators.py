from datetime import datetime

def cont_time(*args,**kwargs):
    def wrapper(func):
        print(args)
        ini = datetime.now()
        func()
        fim = datetime.now()

        print(fim-ini)
        print("acabou")
        
    
    return wrapper

    
