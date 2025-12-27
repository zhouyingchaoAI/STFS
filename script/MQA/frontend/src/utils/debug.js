// 调试工具
export const debugAPI = {
  // 测试API连接
  async testConnection() {
    try {
      const response = await fetch('/api/v1/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: '测试连接',
          options: {}
        })
      })
      
      const data = await response.json()
      console.log('API测试结果:', {
        status: response.status,
        statusText: response.statusText,
        data: data
      })
      
      return {
        success: response.ok,
        status: response.status,
        data: data
      }
    } catch (error) {
      console.error('API连接测试失败:', error)
      return {
        success: false,
        error: error.message
      }
    }
  },
  
  // 测试后端健康检查
  async testHealth() {
    try {
      const response = await fetch('http://localhost:4577/health')
      const data = await response.json()
      console.log('健康检查结果:', data)
      return data
    } catch (error) {
      console.error('健康检查失败:', error)
      return { error: error.message }
    }
  },
  
  // 显示诊断信息
  showDiagnostics() {
    console.group('🔍 前端诊断信息')
    console.log('当前URL:', window.location.href)
    console.log('API Base URL:', '/api/v1')
    console.log('后端地址:', 'http://10.1.6.230:4577')
    console.log('前端地址:', 'http://localhost:3000')
    console.groupEnd()
  }
}

// 在开发环境下自动显示诊断信息
if (import.meta.env.DEV) {
  debugAPI.showDiagnostics()
}
