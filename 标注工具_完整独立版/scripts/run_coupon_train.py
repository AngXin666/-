"""直接运行优惠劵训练"""
import sys
sys.path.insert(0, '.')

# 直接执行训练代码
exec(open('train_coupon_yolo.py', encoding='utf-8').read())

# 调用main函数
if 'main' in dir():
    print("找到main函数，开始执行...")
    main()
else:
    print("未找到main函数")
    print("可用的函数:", [x for x in dir() if not x.startswith('_')])
