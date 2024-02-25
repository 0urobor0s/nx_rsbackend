defmodule NxRSBackend.LinAlg do
  alias NxRSBackend.RS

  # Only support f64
  def qr_binary(tensor) do
    {:f, 64} = Nx.type(tensor)
    {nrow, ncol} = Nx.shape(tensor)

    tensor
    |> Nx.to_binary()
    |> RS.qr_binary(nrow, ncol)
    |> handle_qr(nrow, ncol)
  end

  defp handle_qr({q, r}, nrow, ncol) do
    q = Nx.from_binary(q, :f64) |> Nx.reshape({nrow, ncol})
    r = Nx.from_binary(r, :f64) |> Nx.reshape({ncol, ncol})
    {q, r}
  end

  def qr_tensor(tensor) do
    {:f, 64} = Nx.type(tensor)
    {_nrow, _ncol} = Nx.shape(tensor)

    tensor = convert_to_binary_backend(tensor)
    RS.qr_tensor(tensor)
  end

  defp convert_to_binary_backend(tensor) do
    case tensor.data.__struct__ do
      Nx.BinaryBackend ->
        tensor

      _ ->
        tensor
        |> Nx.to_binary()
        |> Nx.from_binary(:f64, backend: {Nx.BinaryBackend, []})
        |> Nx.reshape(Nx.shape(tensor))
    end
  end
end
