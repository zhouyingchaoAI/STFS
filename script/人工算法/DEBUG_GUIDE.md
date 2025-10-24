# 增长率计算调试指南

## 问题：点击"计算增长率"没有反应

### 调试步骤

#### 1. 打开浏览器开发者工具
- **Chrome/Edge**: 按 `F12` 或 `Ctrl+Shift+I`
- **Firefox**: 按 `F12` 或 `Ctrl+Shift+K`
- 切换到 **Console（控制台）** 标签

#### 2. 清空控制台日志
点击控制台左上角的 🚫 图标清空之前的日志

#### 3. 执行查询和计算
1. 选择指标类型和日期范围
2. 点击"查询数据"
3. 选择对比期日期
4. 点击"计算增长率"

#### 4. 查看控制台输出

应该看到以下日志序列：

```
收到响应: Response { ... }
解析后的数据: { success: true, growth_data: [...], ... }
显示增长率结果: { success: true, growth_data: [...], ... }
增长率数据条数: 4
开始绘制图表, 数据: [...]
图表标签: ['1号线', '2号线', ...]
增长率: [15.23, 8.45, ...]
图表创建成功
```

#### 5. 可能的错误和解决方案

##### 错误 1: `Chart.js 未加载`
**原因**: Chart.js CDN 加载失败
**解决方案**: 
- 检查网络连接
- 更换 Chart.js CDN 源
- 或下载 Chart.js 到本地

##### 错误 2: `计算失败: [错误信息]`
**原因**: 后端计算出错
**解决方案**: 
- 检查选择的日期范围是否有数据
- 查看后端日志（终端）

##### 错误 3: JavaScript 错误
**原因**: 代码执行异常
**解决方案**: 
- 复制完整的错误信息
- 检查是否有拼写错误或语法问题

#### 6. 手动验证数据

在控制台执行以下命令查看数据：

```javascript
// 检查元素是否存在
document.getElementById('growthResultPanel')
document.getElementById('growthChartContainer')
document.getElementById('growthChart')

// 检查 Chart.js 是否加载
typeof Chart !== 'undefined'

// 查看当前查询数据
currentQueryData
```

## 常见问题

### Q: 点击按钮后按钮变灰但没有结果
**A**: 查看控制台是否有错误，可能是网络请求失败或数据格式问题

### Q: 看到"没有找到可比较的数据"
**A**: 
1. 确认基期和对比期都有数据
2. 确认线路编号一致
3. 检查数据库中是否有相应时段的数据

### Q: 图表不显示但卡片显示正常
**A**: 
1. 检查 Chart.js 是否加载（控制台输入 `typeof Chart`）
2. 查看是否有图表相关的错误
3. 尝试刷新页面重新加载

### Q: 所有日志都正常但界面没有变化
**A**: 
1. 检查 CSS 是否正确加载
2. 尝试手动在控制台执行：
```javascript
document.getElementById('growthResultPanel').classList.add('show')
document.getElementById('growthChartContainer').style.display = 'block'
```

## 快速测试

在浏览器控制台粘贴以下代码快速测试：

```javascript
// 测试数据结构
const testData = {
    success: true,
    growth_data: [
        {
            F_LINENO: 1,
            F_LINENAME: '测试线路',
            基期平均: 100000,
            对比期平均: 120000,
            增长率: 20.0,
            增长量: 20000,
            基期天数: 30,
            对比期天数: 5
        }
    ],
    metric_name: '客流量',
    base_period: '2024-04-01 至 2024-04-30',
    compare_period: '2025-05-01 至 2025-05-05'
};

// 调用显示函数
displayGrowthResults(testData);
```

如果这个测试成功显示，说明前端代码正常，问题在于后端返回的数据格式或网络请求。

## 联系支持

如果以上步骤都无法解决问题，请提供：
1. 浏览器控制台的完整错误日志
2. 后端终端的日志输出
3. 使用的浏览器版本
4. 查询的具体日期范围

