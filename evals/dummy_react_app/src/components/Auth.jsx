import { useState } from "react";

// Simple auth component using local state — eval tasks may ask to
// refactor this to use React Context or add form validation.
export default function Auth() {
  const [user, setUser] = useState(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = () => {
    if (!username || !password) {
      setError("Username and password are required");
      return;
    }
    // Hardcoded demo credentials
    if (username === "admin" && password === "password") {
      setUser({ username });
      setError("");
    } else {
      setError("Invalid credentials");
    }
  };

  const handleLogout = () => {
    setUser(null);
    setUsername("");
    setPassword("");
  };

  if (user) {
    return (
      <section>
        <h2>Welcome, {user.username}</h2>
        <button onClick={handleLogout}>Logout</button>
      </section>
    );
  }

  return (
    <section>
      <h2>Login</h2>
      {error && <p style={{ color: "red" }} data-testid="auth-error">{error}</p>}
      <div>
        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          data-testid="username-input"
        />
      </div>
      <div>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          data-testid="password-input"
        />
      </div>
      <button onClick={handleLogin}>Login</button>
    </section>
  );
}

