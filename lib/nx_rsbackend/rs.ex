defmodule NxRSBackend.RS do
  use Rustler, otp_app: :nx_rsbackend, crate: "nxrsbackend"

  # When your NIF is loaded, it will override this function.
  def qr_binary(_binary, _nrow, _ncol), do: :erlang.nif_error(:nif_not_loaded)
  def qr_tensor(_tensor), do: :erlang.nif_error(:nif_not_loaded)
end
