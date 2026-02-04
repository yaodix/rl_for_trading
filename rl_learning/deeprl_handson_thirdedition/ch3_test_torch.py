

import torch


if __name__ == '__main__':
    x = [1.0,2,3]
    t = torch.tensor(x, dtype=torch.bfloat16)
    
    print(t)
    print(t.dtype)
    print(type(t))
    
    t_g = t.to('cuda')
    print(t_g)
    
    ss = torch.FloatTensor([[1,2]])
    print(ss.requires_grad)
    pass