-- Add patient identification + free-text notes to the saved classification rows
-- so reports / history can show who the scan belongs to and the doctor's notes.

alter table public.classifications
  add column if not exists patient_name text,
  add column if not exists patient_id   text,
  add column if not exists notes        text;

-- Allow a row's owner to update its mutable fields (patient_name, patient_id,
-- notes). Strictly enforces that user_id, image_path, probability and
-- prediction can never change after insert.
create policy "Users update own classifications"
on public.classifications for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
