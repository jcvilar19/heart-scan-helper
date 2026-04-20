-- Replace broad public-read policy with one that disallows listing by other users
drop policy if exists "Public read xray-uploads" on storage.objects;

-- Owners can list/read files inside their own folder
create policy "Owners read own folder xray-uploads"
on storage.objects for select
to authenticated
using (
  bucket_id = 'xray-uploads'
  and (storage.foldername(name))[1] = auth.uid()::text
);