/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bull: "#16a34a",
        bear: "#dc2626",
        flat: "#6b7280",
      },
    },
  },
  plugins: [],
};
