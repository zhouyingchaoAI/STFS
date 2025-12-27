import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import './styles/global.css'

import App from './App.vue'
import router from './router'

console.log('main.js: Starting app initialization')

try {
const app = createApp(App)
const pinia = createPinia()

// 注册所有图标
  console.log('main.js: Registering icons')
  try {
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
    }
    console.log('main.js: Icons registered successfully')
  } catch (iconError) {
    console.error('main.js: Icon registration error:', iconError)
}

app.use(pinia)
app.use(router)
app.use(ElementPlus, { locale: zhCn })

  // 添加错误处理
  app.config.errorHandler = (err, instance, info) => {
    console.error('Vue Error:', err, info)
    console.error('Error details:', err.stack)
    // 在页面上显示错误
    const appElement = document.getElementById('app')
    if (appElement) {
      appElement.innerHTML = `
        <div style="padding: 20px; color: red;">
          <h2>应用错误</h2>
          <p>${err.message}</p>
          <pre>${err.stack}</pre>
        </div>
      `
    }
  }

  console.log('main.js: Mounting app to #app')
  const mountElement = document.getElementById('app')
  if (!mountElement) {
    console.error('main.js: #app element not found!')
    document.body.innerHTML = '<div style="padding: 20px; color: red;">错误：找不到 #app 元素</div>'
  } else {
    console.log('main.js: Found #app element:', mountElement)
    try {
app.mount('#app')
      console.log('main.js: App mounted successfully')
    } catch (mountError) {
      console.error('main.js: Mount error:', mountError)
      mountElement.innerHTML = `
        <div style="padding: 20px; color: red;">
          <h2>挂载错误</h2>
          <p>${mountError.message}</p>
          <pre>${mountError.stack}</pre>
        </div>
      `
    }
  }
} catch (error) {
  console.error('main.js: Fatal error:', error)
  const appElement = document.getElementById('app')
  if (appElement) {
    appElement.innerHTML = `
      <div style="padding: 20px; color: red;">
        <h2>初始化错误</h2>
        <p>${error.message}</p>
        <pre>${error.stack}</pre>
      </div>
    `
  }
}

