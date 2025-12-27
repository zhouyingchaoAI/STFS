import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    host: '0.0.0.0',  // 允许外部访问
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:4577',  // 使用localhost（后端在同一台机器）
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path  // 不重写路径
      }
    }
  },
  build: {
    chunkSizeWarningLimit: 1000,  // 提高警告阈值到1MB
    rollupOptions: {
      output: {
        manualChunks: {
          // 将大型库单独打包
          'element-plus': ['element-plus', '@element-plus/icons-vue'],
          'echarts': ['echarts', 'vue-echarts'],
          'vue-vendor': ['vue', 'vue-router', 'pinia']
        }
      }
    }
  }
})

