# 前端项目说明

## 技术栈

- **Vue 3** - 渐进式JavaScript框架
- **Element Plus** - Vue 3 UI组件库
- **ECharts** - 数据可视化图表库
- **Vite** - 下一代前端构建工具
- **Vue Router** - 官方路由管理器
- **Pinia** - 状态管理
- **Axios** - HTTP客户端

## 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
# 或
yarn install
# 或
pnpm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

服务将在 http://localhost:3000 启动

### 3. 构建生产版本

```bash
npm run build
```

构建产物在 `dist` 目录

### 4. 预览生产版本

```bash
npm run preview
```

## 项目结构

```
frontend/
├── src/
│   ├── api/           # API接口
│   ├── views/         # 页面组件
│   ├── components/    # 通用组件
│   ├── router/        # 路由配置
│   ├── App.vue        # 根组件
│   └── main.js        # 入口文件
├── index.html
├── vite.config.js     # Vite配置
└── package.json
```

## 功能特性

- ✅ 美观的现代化UI设计
- ✅ 自然语言查询输入
- ✅ 实时查询结果展示
- ✅ 数据表格展示
- ✅ 自动图表生成
- ✅ 查询历史记录
- ✅ SQL预览
- ✅ 数据导出功能
- ✅ 响应式设计（支持移动端）
- ✅ 深色模式支持

## 配置

### API地址配置

默认API地址为 `http://localhost:8000`，可在 `vite.config.js` 中修改代理配置。

### 主题配置

支持自动跟随系统主题，也可手动切换深色/浅色模式。

## 开发说明

### 添加新页面

1. 在 `src/views/` 创建新组件
2. 在 `src/router/index.js` 添加路由

### 添加新API

在 `src/api/query.js` 中添加新的API方法

## 浏览器支持

- Chrome (推荐)
- Firefox
- Safari
- Edge

## 注意事项

- 确保后端服务运行在 http://localhost:8000
- 如果后端地址不同，需要修改 `vite.config.js` 中的代理配置

