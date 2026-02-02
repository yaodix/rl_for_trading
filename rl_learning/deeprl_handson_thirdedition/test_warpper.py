# 模拟包装器链调用
def simulate_wrapper_chain():
    print("模拟包装器链执行:")
    
    class BaseEnv:
        def reset(self):
            print("  BaseEnv.reset() -> 原始观察")
            return "原始观察", {}
    
    class WrapperA:
        def __init__(self, env):
            self.env = env
        
        def reset(self):
            print("  WrapperA.reset() 开始")
            obs, info = self.env.reset()
            result = f"A({obs})"
            print(f"  WrapperA.reset() 返回: {result}")
            return result, info
    
    class WrapperB:
        def __init__(self, env):
            self.env = env
        
        def reset(self):
            print("  WrapperB.reset() 开始")
            obs, info = self.env.reset()
            result = f"B({obs})"
            print(f"  WrapperB.reset() 返回: {result}")
            return result, info
    
    # 创建包装链
    env = BaseEnv()
    env = WrapperA(env)  # 第一层
    env = WrapperB(env)  # 第二层
    
    print("调用 env.reset():")
    obs, info = env.reset()
    print(f"最终结果: {obs}")
    # 输出: B(A(原始观察))

simulate_wrapper_chain()