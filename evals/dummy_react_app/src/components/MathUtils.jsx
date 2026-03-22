// Utility component that demonstrates math operations.
// NOTE: contains an intentional divide-by-zero bug in safeDivide.

export function safeDivide(a, b) {
  // BUG: does not guard against b === 0
  return a / b;
}

export function factorial(n) {
  if (n < 0) return -1;
  if (n === 0) return 1;
  return n * factorial(n - 1);
}

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export default function MathUtils() {
  return (
    <section>
      <h2>Math Utils</h2>
      <p>10 / 2 = {safeDivide(10, 2)}</p>
      <p>5! = {factorial(5)}</p>
      <p>clamp(15, 0, 10) = {clamp(15, 0, 10)}</p>
    </section>
  );
}

