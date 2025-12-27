import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    console.log('API Request:', config.method?.toUpperCase(), config.url, config.data)
    return config
  },
  error => {
    console.error('Request Error:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    console.log('API Response:', response.status, response.data)
    return response.data
  },
  error => {
    // 处理FastAPI的错误响应格式
    let message = '请求失败'
    let errorDetails = {}
    
    if (error.response) {
      const data = error.response.data
      errorDetails = {
        status: error.response.status,
        statusText: error.response.statusText,
        data: data
      }
      
      // FastAPI错误格式: {"detail": {"code": 500, "message": "...", "data": null}}
      if (data.detail) {
        message = data.detail.message || data.detail.detail || JSON.stringify(data.detail)
      } else if (data.message) {
        message = data.message
      } else {
        message = JSON.stringify(data)
      }
    } else if (error.request) {
      // 请求已发出但没有收到响应
      errorDetails = {
        type: 'network_error',
        message: '网络请求失败，请检查后端服务是否运行'
      }
      message = '网络请求失败，请检查后端服务是否运行'
    } else if (error.message) {
      message = error.message
      errorDetails = { message: error.message }
    }
    
    console.error('API Error Details:', {
      message,
      error: errorDetails,
      config: error.config
    })
    
    return Promise.reject(new Error(message))
  }
)

export const queryAPI = {
  // 自然语言查询
  naturalLanguageQuery(question, options = {}) {
    return api.post('/query', {
      question,
      options
    })
  },
  
  // 自然语言查询（流式版本）
  async naturalLanguageQueryStream(question, options = {}, onMessage = null) {
    try {
      const response = await fetch('/api/v1/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question,
          options
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let messageCount = 0
      
      const startTime = Date.now()
      console.log('[SSE] 开始接收流式数据...', new Date().toISOString())
      
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) {
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(3)
          console.log(`[SSE] 流式数据接收完成，共处理 ${messageCount} 条消息，耗时 ${elapsed}秒`)
          break
        }
        
        // 立即解码并处理
        const decoded = decoder.decode(value, { stream: true })
        if (decoded) {
          buffer += decoded
          const lines = buffer.split('\n')
          buffer = lines.pop() || '' // 保留最后一个不完整的行
          
          for (const line of lines) {
            const trimmedLine = line.trim()
            if (trimmedLine.startsWith('data: ')) {
              try {
                const jsonStr = trimmedLine.slice(6)
                const data = JSON.parse(jsonStr)
                messageCount++
                
                // 立即处理消息，不延迟（确保每个token都立即显示）
                // 对于thinking类型，立即同步处理，不等待任何异步操作
                if (onMessage) {
                  const messageTime = Date.now()
                  if (data.type === 'thinking') {
                    // thinking消息立即同步处理
                    try {
                      onMessage(data)
                      // 调试日志（每个thinking消息都记录，前100个消息详细记录）
                      const elapsed = ((messageTime - startTime) / 1000).toFixed(3)
                      if (messageCount <= 100) {
                        const contentPreview = data.content ? `"${data.content.substring(0, 50)}${data.content.length > 50 ? '...' : ''}"` : 'null'
                        console.log(`[SSE消息 ${messageCount}] (${elapsed}s) type=thinking, length=${data.content?.length || 0}`, contentPreview)
                      } else if (messageCount % 50 === 0) {
                        console.log(`[SSE消息 ${messageCount}] (${elapsed}s) type=thinking, length=${data.content?.length || 0}`)
                      }
                    } catch (e) {
                      console.error('[SSE处理thinking消息错误]', e, data)
                    }
                  } else {
                    // 其他消息正常处理
                    onMessage(data)
                    const elapsed = ((messageTime - startTime) / 1000).toFixed(3)
                    if (messageCount <= 30) {
                      const logContent = JSON.stringify(data).substring(0, 150)
                      console.log(`[SSE消息 ${messageCount}] (${elapsed}s) type=${data.type}`, logContent)
                    }
                  }
                }
              } catch (e) {
                console.error('[SSE解析错误]', e, 'Line:', trimmedLine.substring(0, 100))
              }
            } else if (trimmedLine && !trimmedLine.startsWith(':')) {
              // 忽略注释行，但记录其他非空行
              console.warn('[SSE未知行]', trimmedLine.substring(0, 100))
            }
          }
        }
      }
      
      // 处理剩余的buffer
      if (buffer.trim()) {
        const trimmedBuffer = buffer.trim()
        if (trimmedBuffer.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmedBuffer.slice(6))
            messageCount++
            if (onMessage) {
              onMessage(data)
            }
            console.log(`[SSE最后消息] type=${data.type}`)
          } catch (e) {
            console.error('[SSE] 解析最后buffer失败:', e, trimmedBuffer.substring(0, 100))
          }
        }
      }
    } catch (error) {
      console.error('Stream query error:', error)
      throw error
    }
  },
  
  // SQL直接查询
  sqlQuery(sql, database = 'master') {
    return api.post('/sql', {
      sql,
      database
    })
  },
  
  // 获取查询历史
  getHistory(params = {}) {
    return api.get('/history', { params })
  },
  
  // 获取元数据
  getTables() {
    return api.get('/metadata/tables')
  },
  
  getStations() {
    return api.get('/metadata/stations')
  },
  
  getLines() {
    return api.get('/metadata/lines')
  },
  
  // 健康检查
  healthCheck() {
    return axios.get('/health')
  }
}

export default api

