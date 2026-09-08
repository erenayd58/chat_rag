import { redirect } from 'next/navigation';

/** The console opens on the collection everything else hangs off. */
export default function Home() {
  redirect('/knowledge-bases');
}
