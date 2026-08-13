import type { Metadata } from "next";
import { DemoChrome, DemoHeader } from "../components/chrome";
import { Plane } from "./Plane";

export const metadata: Metadata = {
  title: "The Plane",
  description:
    "Response asymmetry against entropy production, with Sioux Falls, Dominick's, CAISO, rock–paper–scissors and Colonel Blotto as landmarks read from committed artifacts. Drag a game through the plane and watch it leave the line a one-dimensional theory predicts.",
};

export default function Page() {
  return (
    <DemoChrome>
      <DemoHeader
        eyebrow="demo · nowhere else on the web"
        title="The Plane"
        standfirst={
          <p>
            How far a strategic system sits from equilibrium is not one number. It is two, read from two different
            mathematical objects, and they do not agree. This is the plane those two coordinates span, with every system
            anyone has measured so far marked on it — and one quadrant still empty.
          </p>
        }
      />
      <Plane />
    </DemoChrome>
  );
}
