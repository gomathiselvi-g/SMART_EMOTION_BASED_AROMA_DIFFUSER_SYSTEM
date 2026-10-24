importScripts("https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js");

firebase.initializeApp({
  apiKey: "AIzaSyA7FwtVMKvFe_Ub1qZHKiY5MkZ1dM3TdX0",
  authDomain: "smart-aromasense.firebaseapp.com",
  projectId: "smart-aromasense",
  storageBucket: "smart-aromasense.firebasestorage.app",
  messagingSenderId: "32684145893",
  appId: "1:32684145893:web:0829f456de9943f2d5e80c"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage((payload) => {
  const title = payload.notification?.title || "Smart AromaSense";
  const options = {
    body: payload.notification?.body || "You have a new notification."
  };

  self.registration.showNotification(title, options);
});