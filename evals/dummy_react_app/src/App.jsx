import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Header from "./components/Header";
import Counter from "./components/Counter";
import TodoList from "./components/TodoList";
import Auth from "./components/Auth";
import MathUtils from "./components/MathUtils";
import UserProfile from "./components/UserProfile";

export default function App() {
  return (
    <BrowserRouter>
      <Header />
      <nav style={{ padding: "1rem" }}>
        <Link to="/">Home</Link> | <Link to="/todos">Todos</Link> |{" "}
        <Link to="/auth">Auth</Link> | <Link to="/profile">Profile</Link>
      </nav>
      <main style={{ padding: "1rem" }}>
        <Routes>
          <Route
            path="/"
            element={
              <>
                <Counter />
                <MathUtils />
              </>
            }
          />
          <Route path="/todos" element={<TodoList />} />
          <Route path="/auth" element={<Auth />} />
          <Route path="/profile" element={<UserProfile />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

