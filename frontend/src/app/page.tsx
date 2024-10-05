import { Metadata } from "next";
import ChatComponent from "@/components/Chat";
import LazyMap from "@/components/LazyMap";
// import Map from "@/components/Map";

export const metadata: Metadata = {
  title: "IDEA",
};

// const Map = dynamic(() => import("../components/Map"), {
//   loading: () => <p>A map is loading</p>,
//   ssr: false,
// });

export default function Home() {
  return (
    <div className="flex">
      <div className="w-[50rem]">
        <ChatComponent />
      </div>
      <div className="w-[80vw] h-[100vh]">
        <LazyMap />
      </div>
    </div>
  );
}

// export const dynamic = ""
