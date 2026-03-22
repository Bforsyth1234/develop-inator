import { useState, useEffect } from "react";

// Simulates fetching user data. Eval tasks may ask to add loading states,
// error handling, or refactor to a custom hook.
export default function UserProfile() {
  const [profile, setProfile] = useState(null);

  useEffect(() => {
    // Simulate API call
    const timer = setTimeout(() => {
      setProfile({
        name: "Jane Doe",
        email: "jane@example.com",
        role: "Developer",
        joinedAt: "2024-01-15",
      });
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  if (!profile) return <p>Loading profile...</p>;

  return (
    <section>
      <h2>User Profile</h2>
      <dl>
        <dt>Name</dt>
        <dd data-testid="profile-name">{profile.name}</dd>
        <dt>Email</dt>
        <dd data-testid="profile-email">{profile.email}</dd>
        <dt>Role</dt>
        <dd data-testid="profile-role">{profile.role}</dd>
        <dt>Joined</dt>
        <dd data-testid="profile-joined">{profile.joinedAt}</dd>
      </dl>
    </section>
  );
}

