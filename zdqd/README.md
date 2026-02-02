# 自动签到助手

## 项目简介

这是一个用于自动化操作溪盟商城APP的工具，支持多账号管理、自动签到、数据收集等功能。

## 主要功能

- **MuMu模拟器支持**：专为MuMu模拟器优化
- **多账号管理**：支持多个账号同时运行
- **缓存登录**：保存登录状态，无需重复登录
- **自动签到**：每日自动签到领取奖励
- **数据收集**：收集账号信息、余额、积分等数据
- **数据库存储**：使用SQLite数据库管理历史记录
- **卡密激活**：支持在线激活和许可证管理
- **GUI界面**：友好的图形界面操作

## 快速开始

### 1. 直接运行（推荐）

```bash
# 运行主程序
python run.py

# 运行卡密管理工具
python 卡密管理GUI.py
```

### 2. 使用打包后的EXE

双击 `dist` 目录中的可执行文件：
- `溪盟自动化.exe` - 主程序
- `卡密管理.exe` - 卡密管理工具

## 配置文件

配置文件位于 `config` 目录：
- `config.yaml` - 主配置文件（模拟器设置、账号文件路径等）
- `transfer_config.json` - 转账配置
- `database_setup.sql` - 数据库初始化脚本

## 构建可执行文件

```bash
# 进入构建脚本目录
cd build_scripts

# 构建主程序
build_exe.bat

# 构建安装包
build_installer.bat

# 构建卡密管理工具
build_license_manager.bat
```

构建完成后，可执行文件位于 `dist` 目录。

## 项目结构

```
├── src/                    # 源代码目录
│   ├── gui.py             # GUI界面
│   ├── orchestrator.py    # 主控制器
│   ├── ximeng_automation.py # 自动化逻辑
│   ├── auto_login.py      # 登录模块
│   ├── login_cache_manager.py # 登录缓存管理
│   ├── daily_checkin.py   # 签到模块
│   ├── checkin_page_reader.py # 签到页面读取
│   ├── profile_reader.py  # 个人信息模块
│   ├── account_cache.py   # 账号缓存
│   ├── navigator.py       # 页面导航
│   ├── page_detector*.py  # 页面检测
│   ├── balance_reader.py  # 余额读取
│   └── performance/       # 性能优化模块
├── docs/                   # 文档目录
│   ├── 开发文档/          # 开发相关文档
│   ├── 测试文档/          # 测试相关文档
│   ├── 优化报告/          # 性能优化报告
│   └── 历史版本/          # 历史版本记录
├── dist/                   # 构建输出目录
├── logs/                   # 日志目录
├── reports/                # 报告目录
├── config.yaml            # 主配置文件
└── requirements.txt       # Python依赖

```

## 技术栈

- **Python 3.x**
- **PyQt5** - GUI界面
- **OpenCV** - 图像处理
- **PaddleOCR** - 文字识别
- **ADB** - Android调试桥
- **PyInstaller** - 打包工具

## 文档

### 主要文档
- `更新日志.md` - 版本更新历史
- `docs/README.md` - 文档索引

### 文档分类
- `docs/开发文档/` - 项目结构、构建说明、模板加密等
- `docs/测试文档/` - 测试指南、测试报告
- `docs/优化报告/` - 性能优化分析和报告
- `docs/历史版本/` - 历史版本记录和整理总结

## 版本历史

详见 `更新日志.md`

当前版本：**v1.7.1** - 缓存登录积分页跳转修复

## 注意事项

1. 首次运行需要安装模拟器（夜神/雷电/MuMu）
1. 配置 `config.yaml` 文件
2. 确保模拟器已安装溪盟商城APP
3. 建议使用管理员权限运行

## 许可证

本项目仅供学习交流使用。
