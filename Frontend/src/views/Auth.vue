<template>
  <div class="container">
    <header>
      <img
        src="@/assets/COCOMOOD_logo_white.png"
        alt="Cocomood logo"
        class="logo"
      />
      <router-link to="/reg" class="login-btn">Регистрация</router-link>
    </header>
    <div class="registration-box">
      <h2>Войти</h2>
      <form @submit.prevent="register">
        <div class="input-group">
          <label for="login">Логин</label>
          <input type="text" id="login" v-model="login" />
        </div>
        <div class="input-group">
          <label for="password">Пароль</label>
          <input
            :type="showPassword ? 'text' : 'password'"
            id="password"
            v-model="password"
          />
          <!-- <div class="eye" @click="togglePasswordVisibility">👁️</div> -->
        </div>
        <button type="submit" class="register-btn" @click="log_in">
          <!-- <router-link to="/tracker1" class="btn_btn" @click="log_in" -->
          <!-- >Войти</router-link -->
          > Войти!!!!
        </button>
      </form>
    </div>
  </div>
</template>

<script>
import { useMainStore } from '@/stores/store'
import { storeToRefs } from 'pinia'
import { computed, watch } from 'vue'
import axios from 'axios'
export default {
  data() {
    return {
      name: '',
      login: '',
      password: '',
      showPassword: false,
    }
  },
  setup() {
    const mainStore = useMainStore()
    const { isAuthorized } = storeToRefs(useMainStore)

    return { mainStore, isAuthorized }
  },
  methods: {
    togglePasswordVisibility() {
      this.showPassword = !this.showPassword
    },
    log_in() {
      // this.login({ login: this.login, password: this.password })
      this.mainStore.login({ login: this.login, password: this.password })
    },
  },
}
</script>

<style scoped>
.container {
  font-family: Arial, sans-serif;
  text-align: center;
  padding: 20px;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: #00a195;
}

.logo {
  height: 40px;
}

.login-btn {
  padding: 8px 16px;
  background: white;
  border: 1px solid #333;
  cursor: pointer;
  color: #333;
}

.registration-box {
  color: #333;
  background: white;
  padding: 30px;
  margin: 50px auto;
  max-width: 300px;
  border-radius: 8px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

h2 {
  font-size: 24px;
  margin-bottom: 20px;
}

.input-group {
  margin-bottom: 15px;
  text-align: left;
}

input[type='text'],
input[type='password'] {
  width: 100%;
  padding: 10px;
  margin-top: 5px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.register-btn {
  width: 100%;
  padding: 10px;
  background-color: #333;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 16px;
}

span {
  cursor: pointer;
  position: absolute;
  right: 15px;
  top: 38px;
  font-size: 18px;
}

.eye {
  float: right;
}

.btn_btn {
  color: #ddd;
}
</style>
